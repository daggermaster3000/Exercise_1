#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date: May 2018
Authors: Alexandra Gastone, Moritz Gruber, Patrick Haller
COMPUTER SIMULARIONS OF SENSORY SYSTEMS -- VESTIBULAR IMPLANTS
INSTRUCTIONS 
    // Place this file in the same directory as the folder MovementData containing the file 
       'Walking_02.txt'.
    // Make sure the module vpython is installed. If need be, open a terminal window and 
       enter 'pip install vpython'.
    // To run the program, open a terminal window and run 'python Vestibular.py'
    // A browser window will open to show the visualization of the head orientation over time
    // The program writes four files to the directory it is executed from:
            1. 'CupularDisplacement.txt' containing the maximum cupular displacements 
            2. 'MaxAcceleration.txt'     containing the maximum acceleration along the 
                                         polarization vector of a given otolith cell
            3. 'FinalNose.txt'           containing the final nose orientation
            4. 'HeadOrientation.png'     a plot showing the head orientation over time in terms
                                         of quaternions
"""

import os
import skinematics as skin
import numpy as np
from skinematics.sensors.xsens import XSens
import scipy.signal as sig
import matplotlib.pyplot as plt
from vpython import *

def main():
    """
    Main function, reads in provided walking data and calculates
        - maximum cupular displacements (positive and negative) in the right horizontal SCC
        - minimum and maximum acceleration along a given otolith hair cell
        - final "nose vector" 
    """

    # read in data
    abspath = os.path.abspath(__file__) 
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    in_file = 'C:\\Users\\quill\\Documents\\UZH-2\\Sensory-systems\\Exercises\\Exercise_2\\MovementData\\Walking_01.txt'
    acceleration, angular_velocity = get_sensor_data(in_file)
    sensor_reference = acceleration[0]
    ideal_gravity = np.r_[0,9.81,0]

    # maximum cupular displacements
    updated_angular_velocity = align_to_sensor(
        angular_velocity,sensor_reference,ideal_gravity)
    np.savetxt("updated_angular_them.txt",updated_angular_velocity)
    deflection = get_cupular_displacement(updated_angular_velocity)
    print('Cupular deflection', deflection)
    
    np.savetxt("CupularDisplacement.txt",deflection,fmt='%10.5f')

    # minimum and maximum acceleration
    updated_acceleration_data = align_to_sensor(
        acceleration,sensor_reference,ideal_gravity)
    np.savetxt("updated_acc_them.txt",updated_acceleration_data)
    acceleration = get_otolith_stimulation(updated_acceleration_data)
    print('Acceleration', acceleration)
    np.savetxt("MaxAcceleration.txt",acceleration,fmt='%10.5f')

    # get head and nose direction during the walk
    head, nose = get_nose_orientation(updated_angular_velocity)
    print('Final nose orientation', nose[-1])
    np.savetxt("FinalNose.txt",nose[-1],fmt='%10.5f')

    # plot head orientation
    t = np.arange(0,len(angular_velocity))/50
    plt.plot(t, head[:,1:4])
    plt.xlabel('Time [s]')
    plt.ylabel('Head orientation [quat]')
    plt.savefig('HeadOrientation')

    # visualize nose direction during walking
    visualize(nose)

def get_sensor_data(file):
    """
    Returns acceleration and gyrodata recorded by IMU
    """
    data = XSens(file, q_type=None)
    return data.acc, data.omega


def align_to_sensor(data, sensor_reference, ideal_gravity):
    """
    Aligns sensor data wrt space (project on global, space-fixed axis and correct for gravity)
    """
    quaternion_adjust = skin.vector.q_shortest_rotation(sensor_reference, ideal_gravity) # This is the quaternion we have to rotate our sensor data about.
    quaternion_rot= np.r_[np.sin(np.pi/4),0, 0] #describes 90deg rotation to the right (negative)
    quaternion_total = skin.quat.q_mult(quaternion_rot, quaternion_adjust)

    adjusted_data = skin.vector.rotate_vector(data,quaternion_total)
    return adjusted_data


def get_cupular_displacement(gyrodata):
    """"
    Calculate cupular displacement in right SCC based on gyrodata
    """
    # define radius of SCC
    radius_canal = 3.2
    CANALS = {'info': 'The matrix rows describe horizontal, anterior, and posterior canal orientation',
    'right': np.array(
           [[0.365,  0.158, -0.905],
            [0.652,  0.753, -0.017],
            [0.757, -0.561,  0.320]]),
    'left': np.array(
           [[-0.365,  0.158,  0.905],
            [-0.652,  0.753,  0.017],
            [-0.757, -0.561, -0.320]])}

     # normalize these vectors
    for side in ['right', 'left']:
         CANALS[side] = (CANALS[side].T / np.sqrt(np.sum(CANALS[side]**2, axis=1))).T 

    right_horizontal_canal = CANALS['right'][0]
    # calculate the stimulation as dot product of gyrodata and right SCC orientation
    stimulation = gyrodata @ right_horizontal_canal

    # get transferfunction (mechanical response of the canal)
    # define constants
    T1 = 0.01
    T2 = 5.0
    numerator = [T1*T2,0] 
    denominator = [T1*T2,T1+T2,1] 
    system = sig.lti(numerator,denominator)

    # take into account frequency component (50 Hz) for time axis
    t = np.arange(0,len(gyrodata))/50

    # cupula deflection
    _, outSignal, _ = sig.lsim(system,stimulation, t)
    deflection = outSignal * radius_canal
    plt.plot(deflection)
    plt.show()
    min_deflection = np.min(deflection) # mm
    max_deflection = np.max(deflection) # mm
    deflection = np.array([min_deflection, max_deflection])
    return deflection

def get_otolith_stimulation(acceleration_data):
    """
    Stimulation of otolith hair cell pointing to the left
    """
    # acceleration along [0, 1, 0]
    y_acceleration = []
    for i in range(len(acceleration_data)):
        y = acceleration_data[i][1]
        y_acceleration.append(y)

    max_y = np.max(y_acceleration)
    min_y = np.min(y_acceleration)

    acceleration = np.array([min_y, max_y])
    return acceleration

def get_nose_orientation(angular_velocity):
    """
    Head and nose orientation during movement
    """
    # get to head oriented coordinates, rotation about y axis 15 degrees down 
    rot = skin.rotmat.R(2,15)
    rotated_angular_velocity = []
    for i in range(len(angular_velocity)):
        rot_data = np.matmul(rot, angular_velocity[i])
        rotated_angular_velocity.append(rot_data)

    head_orientation = skin.quat.calc_quat(angular_velocity, [0,0,0], rate=50, CStype='bf')

    nose_rotmat_all = [skin.quat.convert(i, to='rotmat') for i in head_orientation]
    nose_all = [np.matmul(i,np.array([1,0,0])) for i in nose_rotmat_all]

    return head_orientation, nose_all

def visualize(nose):
    """
    Visualization of the nose movemente during the walk. Opens a browser window.
    """
    # draw coordinate system
    cs = curve(pos=[(-2, 0, 0), (2, 0, 0), 
                           (0, 0, 0), (0, 2, 0), (0, -2, 0),
                            (0,0,0), (0, 0, 2), (0,0,-2)])
    horPlane = curve(pos=[(-2,0,-2),(-2,0,2),(2,0,2),(2,0,-2),(-2,0,-2)])
        
    xLabel = label(pos=vec(0,0,2), text='X')
    yLabel = label(pos=vec(2,0,0), text='Y')
    zLabel = label(pos=vec(0,2,0), text='Z')
    ii = 0
    q_arrow = arrow(pos=vec(0,0,0), 
                axis=vec(0,0,1),
                shaftwidth=0.2)

    # visualize movement
    while ii<len(nose)-1:
            ii = ii+1
            rate(50)
            dir_vector = nose[ii]
            q_arrow.axis=vec(dir_vector[0], dir_vector[2], dir_vector[1])

if __name__ == '__main__':
    main()


"""
Exercise: Simulation of a Vestibular Implant
Authors: Quillan Favey, Alessandro Pasini
Version: 1.0
Date: 25.04.2023

This file contains code for simulating the output of a 
vestibular implant using data from an IMU.


Naming convention in this file: 

Rotation matricies start with R
quaternions start with q
R_a_b is rotation from coordinates a to coordinates b
name_a is a vector in coordinates a
approx: approximate IMU coordinates
IMU: IMU coordinates (all measurements are in these coordinates)
hc: head coordinates / 'world coordinates at t=0'
rc: Reid's line coords


TODO:
MISC
- adjusted means relative to the head
- write into functions
- loading bars for each function
STEPS
- 1 <done>
- 2 <done>
- 3 <done>
- 4 <done>
- 5
- 6
- 7 
- 8
- 9
...



"""

# Import libraries
import numpy as np
import skinematics as skin
from skinematics.sensors.xsens import XSens
import scipy.io as sio




# Functions

def q_shortest(a,b):
    """
    INPUTS
    a: a vector to be brought parralell to b
    b: a vector
    ----------
    returns: q_shortest, the shortest rotation that brings a to b
    """
    n =  np.cross(a,b)/np.linalg.norm(np.cross(a,b))
    alpha = np.arccos(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
    q_shortest = n * np.sin(alpha/2)
    return q_shortest

def get_data_sensor(file_path):
    """
    INPUTS
    file_path: the path to the file containing the sensor data
    ----------
    returns: Acceleration and angular velocities measured by the sensor
    """
    data = XSens(file_path,q_type=None)
    return data.acc, data.omega

def align_sensor_vectors():
    ...

#1. Read in the data (use only the 3D-acceleration and the 3D-angular-velocity! I expect you to calculate the orientation-quaternion yourself!) 

# Read data from IMU
in_file = 'Exercise_2\MovementData\Walking_01.txt'
acc, angular_vel = get_data_sensor(in_file)

#2. Find q˜0, i.e. the orientation of the IMU at t=0 re space 

# Set in head coordinates the accelerations sensed by the IMU 
a_IMU_start_hc = np.r_[0,0,-9.81]

# Convert to sensor coordinates the "shortest rotation that aligns 
# the y-axis of the sensor with gravity" brings the sensor into such 
# an orientation that the (x/ -z / y) axes of the sensor aligns with the space-fixed (x/y/z) axes
# So a 90 deg rotation around the x-axis"
q_rotate = np.r_[np.sin(np.deg2rad(90)/2), 0, 0]
a_IMU_ref_sc = np.r_[0,9.81,0]

# Next we can get the data from the sensor at t=0 and compute the shortest quaternion going from this vector
# to the sensor coordinate vectors (assuming the only acceleration at t=0 is gravity). View p.67 of 3D-Kinematics 
# for details
a_t0 = acc[0,:]
q_adjust = q_shortest(a_t0,a_IMU_ref_sc)
q_total = skin.quat.q_mult(q_rotate,q_adjust)
# Adjust all the data
acc_adjusted = skin.vector.rotate_vector(acc,q_total)
angular_vel_adjusted = skin.vector.rotate_vector(angular_vel,q_total)


#3. Find n0, i.e. the orientation of the right SCC (semicircular canal) at t=0 
# Define the measured orientations of the SCCs relative to Reid's plane (from the IPYNBs)

Canals = {
    'info': 'The matrix rows describe ' +\
            'horizontal, anterior, and posterior canal orientation',
    'right': np.array(
        [[0.32269, -0.03837, -0.94573], 
         [0.58930,  0.78839,  0.17655],
         [0.69432, -0.66693,  0.27042]]),
    'left': np.array(
        [[-0.32269, -0.03837, 0.94573], 
         [-0.58930,  0.78839,  -0.17655],
         [-0.69432, -0.66693,  -0.27042]])}

# Normalize these vectors (just a small correction):
for side in ['right', 'left']:
    Canals[side] = (Canals[side].T / np.sqrt(np.sum(Canals[side]**2, axis=1))).T

# now we can adjust n0 to the head coordinates
R_rot = skin.rotmat('y',15)
Right_horizontal_SCC = Canals['right'][0]
n0_HC = R_rot.dot(Right_horizontal_SCC.T).T

#4. Using q˜0, ⃗n0 and ⃗ω(t) sensor , calculate stim, the stimulation of the right SCC 
stim = angular_vel_adjusted @ n0_HC

#5. Find the canal-transfer-function re velocity (not re rotation-angle!) 


#6. Using stim and the canal-transfer-function, calculate the cupula deflection 


#7. Using q˜0 and ⃗ω(t) sensor , calculate q˜(t), i.e. the head orientation re space during the movement 


#8. Calculate the stimulation of the otolith hair cell 


#9. Show the head orientation, expressed with quaternions as a function of time

print('done')
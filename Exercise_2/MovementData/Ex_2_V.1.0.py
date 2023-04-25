
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
"""

# Import libraries
import numpy as np
import skinematics as skin
from skinematics.sensors.xsens import XSens
import time




# Functions

def q_shortest(a,b):
    """
    a: a vector to be brought parralell to b
    b: a vector
    returns: q_shortest the shortest rotation that brings a to b
    """
    n =  np.cross(a,b)/np.norm(np.cross(a,b))
    alpha = np.arcos(np.dot(a,b)/(np.norm(a)*np.norm(b)))
    q_shortest = n * np.sin(alpha/2)
    return skin.quat.quaternion(q_shortest)

#1. Read in the data (use only the 3D-acceleration and the 3D-angular-velocity! I expect you to calculate the orientation-quaternion yourself!) 
# Read data
data = XSens(in_file='Exercise_2\MovementData\Walking_01.txt',q_type=None)
print(data.totalSamples)
# Store accelerations from the IMU
accelerations = data.acc


#2. Find q˜0, i.e. the orientation of the IMU at t=0 re space 

# Set in head coordinates the accelerations sensed by the IMU 
a_IMU_start_hc = np.array([0,0,-9.81])

# Convert to sensor coordinates the "shortest rotation that aligns 
# the y-axis of the sensor with gravity" brings the sensor into such 
# an orientation that the (x/ -z / y) axes of the sensor aligns with the space-fixed (x/y/z) axes
# So a 90 deg rotation around the x-axis
q_adjust = [np.sin(np.deg2rad(90)/2), 0, 0]
a_IMU_start_sc = skin.vector.rotate_vector(a_IMU_start_hc, q_adjust)
print(a_IMU_start_sc)

# Next we can get the data from the sensor at t=0 and compute the shortest quaternion going from this vector
# to the sensor coordinate vectors (assuming the only acceleration at t=0 is gravity). View p.67 of 3D-Kinematics 
# for details
a_t0 = accelerations[0,:]

alpha = q_shortest(a_IMU_start_sc,a_t0)


# Compute the shortest rotation from the IMU's base position to 




#3. Find n0, i.e. the orientation of the right SCC (semicircular canal) at t=0 


#4. Using q˜0, ⃗n0 and ⃗ω(t) sensor , calculate stim, the stimulation of the right SCC 


#5. Find the canal-transfer-function re velocity (not re rotation-angle!) 


#6. Using stim and the canal-transfer-function, calculate the cupula deflection 


#7. Using q˜0 and ⃗ω(t) sensor , calculate q˜(t), i.e. the head orientation re space during the movement 


#8. Calculate the stimulation of the otolith hair cell 


#9. Show the head orientation, expressed with quaternions as a function of time


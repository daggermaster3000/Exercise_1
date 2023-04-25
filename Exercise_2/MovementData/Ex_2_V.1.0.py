
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


#1. Read in the data (use only the 3D-acceleration and the 3D-angular-velocity! I expect you to calculate the orientation-quaternion yourself!) 



#2. Find q˜0, i.e. the orientation of the IMU at t=0 re space 



#3. Find ⃗n0, i.e. the orientation of the right SCC (semicircular canal) at t=0 


#4. Using q˜0, ⃗n0 and ⃗ω(t) sensor , calculate stim, the stimulation of the right SCC 


#5. Find the canal-transfer-function re velocity (not re rotation-angle!) 


#6. Using stim and the canal-transfer-function, calculate the cupula deflection 


#7. Using q˜0 and ⃗ω(t) sensor , calculate q˜(t), i.e. the head orientation re space during the movement 


#8. Calculate the stimulation of the otolith hair cell 


#9. Show the head orientation, expressed with quaternions as a function of time



"""
Exercise: Simulation of a Vestibular Implant
Authors: Quillan Favey, Alessandro Pasini
Version: 1.1
Date: 25.04.2023

This file contains code for simulating the output of a 
vestibular implant using data from an IMU.


Naming convention in this file: 

Rotation matrices start with R
quaternions start with q
R_a_b is rotation from coordinates a to coordinates b
name_a is a vector in coordinates a
IMU: IMU coordinates (all measurements are in these coordinates)
hc: head coordinates / 'world coordinates at t=0'
rc: Reid's line coords


TODO:
MISC
- adjusted means relative to the head
- write as functions
- loading bars for each function
STEPS
- 1 <done>
- 2 <done>
- 3 <done>
- 4 <done>
- 5 <done>
- 6 <done>
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
from scipy import signal
import matplotlib.pyplot as plt
import time
import sys
import threading
import itertools

# Functions

def running_decorator(func):
    def wrapper(*args, **kwargs):
        msg = f"Running {func.__name__}..."
        print(msg, end="\r\t\t\t\t\t")
        sys.stdout.flush()
        event = threading.Event()
        loading_thread = threading.Thread(target=print_loading_message, args=(event,msg))
        
        loading_thread.start()
        
        result = func(*args, **kwargs)
        
        event.set()

        loading_thread.join()
        

        return result

    def print_loading_message(event,msg):
        for char in itertools.cycle("|/-\\"):
            if event.is_set():
                print("Done.")
                break
            print(char, end="\r\t\t\t\t\t")
            sys.stdout.flush()
            

    return wrapper


def q_shortest(a, b):
    """
    INPUTS:
    a: a vector to be brought parralell to b
    b: a vector
    ----------
    Returns: q_shortest, the shortest rotation (quaternion) that brings a to b
    """
    n = np.cross(a, b)/np.linalg.norm(np.cross(a, b))
    alpha = np.arccos(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))
    q_shortest = n * np.sin(alpha/2)
    return q_shortest

@running_decorator
def get_data_sensor(file_path):
    """
    INPUTS:
    file_path: the path to the file containing the sensor data
    ----------
    Returns: Acceleration and angular velocities measured by the sensor
    """
    data = XSens(file_path, q_type=None)
    return data.acc, data.omega

@running_decorator
def align_sensor_vectors(a_IMU_base_IC, a_IMU_head_HC, a_IMU_t0, acc, omegas):
    """
    Define and compute the necessary rotations to go back to head coordinates (HC) from sensor coordinates (IC) (elaborate)
    ----------
    INPUTS:
    a_IMU_base_IC:   The vector acting on the IMU in it's original position (in IMU coordinates  (IC))
    a_IMU_head_IC:   The vector acting on the IMU when fixed to the head (in IMU coordinates (IC))
    a_IMU_t0:        The vector acting on the IMU at t=0
    acc:             The data containing the accelerations
    omegas:          The data containing the angular velocities
    ----------
    Returns: Acceleration and angular velocities rotated into head coordinates (elaborate)
    """
    # Convert to sensor coordinates the "shortest rotation that aligns
    # the y-axis of the sensor with gravity" brings the sensor into such
    # an orientation that the (x/ -z / y) axes of the sensor aligns with the space-fixed (x/y/z) axes
    # So a 90 deg rotation around the x-axis"
    q_rotate = q_shortest(a_IMU_base_IC, a_IMU_head_HC)

    # Next we can get the data from the sensor at t=0 and compute the shortest quaternion going from this vector
    # to the sensor coordinate vectors (assuming the only acceleration at t=0 is gravity). View p.67 of 3D-Kinematics
    # for details
    q_adjust = q_shortest(a_IMU_t0, a_IMU_head_IC)
    q_total = skin.quat.q_mult(q_rotate, q_adjust)
    # Adjust all the data
    acc_adjusted = skin.vector.rotate_vector(acc, q_total)
    omegas_adjusted = skin.vector.rotate_vector(omegas, q_total)

    return acc_adjusted, omegas_adjusted

@running_decorator
def get_cupular_deflections(num, den, t, stim):
    """
    Compute the cupula deflections from stimulation data according to the transfer function.
    ----------
    INPUTS:
    num:    Numerator of the transfer function
    den:    Denominator of the transfer function
    t:      Time array
    stim:   Stimulation data
    ----------
    Returns: An array containing the cupular deflections as a function of time
    """
    time.sleep(10)
    tf_canals = signal.lti(num, den)  # Create transfer function

    # 6. Using stim and the canal-transfer-function, calculate the cupula deflection
    # Simulate and return cupular deflections
    t_out_cupular, cupular_deflection, state_vector = signal.lsim(
        tf_canals, stim, t)  # state vector is no needed
    return cupular_deflection





# 1. Read in the data (use only the 3D-acceleration and the 3D-angular-velocity! I expect you to calculate the orientation-quaternion yourself!)


# Read data from IMU
in_file = 'Exercise_2\MovementData\Walking_01.txt'
acc, omega = get_data_sensor(in_file)

# 2. Find q˜0, i.e. the orientation of the IMU at t=0 re space

# Set  sensed by the IMU
a_IMU_base_IC = np.r_[0, 0, -9.81]
a_IMU_head_IC = np.r_[0, 9.81, 0]
a_IMU_t0 = acc[0, :]

acc_adjusted, omegas_adjusted = align_sensor_vectors(
    a_IMU_base_IC, a_IMU_head_IC, a_IMU_t0, acc, omega)
# np.savetxt("accs_v2.txt",acc_adjusted)

# 3. Find n0, i.e. the orientation of the right SCC (semicircular canal) at t=0

# Define the measured orientations of the SCCs relative to Reid's plane (from the IPYNBs)

Canals = {
    'info': 'The matrix rows describe ' +
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
    Canals[side] = (Canals[side].T /
                    np.sqrt(np.sum(Canals[side]**2, axis=1))).T

# now we can adjust n0 to the head coordinates
R_rot = skin.rotmat.R('y', 15)
Right_horizontal_SCC = Canals['right'][0]
n0_HC = R_rot.dot(Right_horizontal_SCC.T).T

# 4. Using q˜0, ⃗n0 and ⃗ω(t) sensor , calculate stim, the stimulation of the right SCC
stim = omegas_adjusted @ n0_HC

# 5. Find the canal-transfer-function re velocity (not re rotation-angle!)

T1 = 0.01  # Define time constants [s]
T2 = 5
num = [T1*T2, 0]  # Define numerator
den = [T1*T2, T1+T2, 1]  # Define denominator
t = np.arange(0, len(stim))/50  # Define time array (from 50 Hz sample rate)
cupula_defl = get_cupular_deflections(num, den, t, stim)    # simulate and get cupular deflections

# convert to mm
r_canal = 3.2
cupula_defl = cupula_defl*r_canal
np.savetxt("Cupula_deflections.txt", [
           np.min(cupula_defl), np.max(cupula_defl)],fmt='%10.5f')   # save to .txt file


# 7. Using q˜0 and ⃗ω(t) sensor , calculate q˜(t), i.e. the head orientation re space during the movement
#acc_reHead = R_total * acc_reSensor

#acc_sensed = acc_reHead * n_otolith

# 8. Calculate the stimulation of the otolith hair cell


# 9. Show the head orientation, expressed with quaternions as a function of time

print('done')

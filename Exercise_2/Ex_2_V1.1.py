
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
- adjusted means relative to the head <done>
- write as functions <done>
- loading wheels for each function <done> (sort of doesnt work yet on conda prompt)
- 3D animation 
- Propre
STEPS
- 1 <done>
- 2 <done>
- 3 <done>
- 4 <done>
- 5 <done>
- 6 <done>
- 7 head orientation in space <done>
- 8 otolith hair cell simulation <done>
- 9 show head orientation with quaternions over time <done>
...

"""

# Import libraries import only the functions we need for faster loading

from skinematics.sensors.xsens import XSens
from skinematics.quat import q_mult, calc_quat, convert
from skinematics.vector import rotate_vector
from skinematics.rotmat import R as rotmat
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import threading
import itertools
import os
import webbrowser

# Main function

def main():
    
    # 1. Read in the data (use only the 3D-acceleration and the 3D-angular-velocity! I expect you to calculate the orientation-quaternion yourself!)


    # Read data from IMU
    in_file = 'Exercise_2\MovementData\Walking_02.txt'
    acc, omega, rate, n_samples = get_data_sensor(in_file)

    # 2. Find q˜0, i.e. the orientation of the IMU at t=0 re space

    # Set accelerations sensed by the IMU
    a_IMU_base_IC = np.r_[0, 0, -9.81]
    a_IMU_head_IC = np.r_[0, 9.81, 0]
    a_IMU_t0 = acc[0, :]

    acc_adjusted, omegas_adjusted, q_0 = align_sensor_vectors(
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
    R_rot = rotmat('y', 15)
    Right_horizontal_SCC = Canals['right'][0]
    n0_HC = R_rot.dot(Right_horizontal_SCC.T).T

    # 4. Using q˜0, ⃗n0 and ⃗ω(t) sensor , calculate stim, the stimulation of the right SCC
    stim = omegas_adjusted @ n0_HC

    # 5. Find the canal-transfer-function re velocity (not re rotation-angle!)

    T1 = 0.01  # Define time constants [s]
    T2 = 5
    num = [T1*T2, 0]  # Define numerator
    den = [T1*T2, T1+T2, 1]  # Define denominator
    t = np.arange(0, n_samples)/rate  # Define time array (from 50 Hz sample rate)
    cupula_defl = get_cupular_deflections(num, den, t, stim)    # simulate and get cupular deflections

    # convert to mm
    r_canal = 3.2
    cupula_defl = cupula_defl*r_canal
    np.savetxt("Exercise_2\Outputs\Cupula_deflections.txt", [
            np.min(cupula_defl), np.max(cupula_defl)],fmt='%10.5f')   # save to .txt file


    # 7. Using q˜0 and ⃗ω(t) sensor , calculate q˜(t), i.e. the head orientation re space during the movement
    head_orientation_v, head_orientation_q = calculate_head_orientation(omegas_adjusted)

    # 8. Calculate the stimulation of the otolith hair cell
    stim_otolith = stim_otolith_left(acc_adjusted)
    # write to file
    ...


    # 9. Show the head orientation, expressed with quaternions as a function of time
    show_head_orientation(...,head_orientation_q,head_orientation_v,t)

    print('Done!')





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
            time.sleep(0.1)
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
    Returns: Acceleration, angular velocities, rate and total samples measured by the sensor
    """
    data = XSens(file_path, q_type=None)
    return data.acc, data.omega, data.rate, data.totalSamples

@running_decorator
def align_sensor_vectors(a_IMU_base_IC, a_IMU_head_IC, a_IMU_t0, acc, omegas):
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
    q_rotate = q_shortest(a_IMU_base_IC, a_IMU_head_IC)
    
    # Next we can get the data from the sensor at t=0 and compute the shortest quaternion going from this vector
    # to the sensor coordinate vectors (assuming the only acceleration at t=0 is gravity). View p.67 of 3D-Kinematics
    # for details
    q_adjust = q_shortest(a_IMU_t0, a_IMU_head_IC)
    q_total = q_mult(q_rotate, q_adjust)
    # Adjust all the data
    acc_adjusted = rotate_vector(acc, q_total)
    omegas_adjusted = rotate_vector(omegas, q_total)

    return acc_adjusted, omegas_adjusted, q_total

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
    
    tf_canals = signal.lti(num, den)  # Create transfer function

    # 6. Using stim and the canal-transfer-function, calculate the cupula deflection
    # Simulate and return cupular deflections
    t_out_cupular, cupular_deflection, state_vector = signal.lsim(
        tf_canals, stim, t)  # state vector is no needed
    return cupular_deflection

@running_decorator
def calculate_head_orientation(adjusted_omegas,nose_init=np.r_[1,0,0]):
    """
    Calculate the head/nose orientation re space during the movement
    ----------
    INPUTS:
    adjusted_omegas:    The adjusted angular velocities 
    nose_init:          The initial orientation of the nose    
    ----------
    Returns: 
    head_orientation_vect:  An array containing the head/nose orientation 
    head_orientation_quat:  The quaternions describing the rotations during the walk
    """
    # Convert angular velocities to quaternion
    head_orientation_quat = calc_quat(adjusted_omegas, [0,0,0], rate=50, CStype='bf')

    # Convert to matrix
    rotmat = []
    for i in head_orientation_quat:
        rotmat.append(convert(i, to='rotmat'))

    # Rotate the initial position vectors according to the rotation matrix to get the orientation
    head_orientation_vect = [np.matmul(i,nose_init) for i in rotmat]

    return head_orientation_vect, head_orientation_quat


@running_decorator
def stim_otolith_left(adjusted_accelerations):
    """
    Simulate the stimulation of an otolith hair cell pointing to the left [0, 1, 0]
    (just acceleration vectors pointing to the left)
    ----------
    INPUTS:
    adjusted_acceleration:      An array containing adjusted acceleration data (in HC)
    ----------
    Returns: y_acc, An array containing the left polarized otolith hair cell's stimulation as a function of time
    """
    y_acc = []
    for i in range(0,len(adjusted_accelerations)):
        y_acc.append(adjusted_accelerations[i][1])
    return y_acc

@running_decorator
def show_head_orientation(path,head_orientation_q,head_orientation_v,t):
    """
    Creates a .avi file playing the nose orientation as a function of time or maybe 
    a .js file that will be opened in the browser
    ----------
    INPUTS:
    ...
    ----------
    Returns: Void
    """

    # A bit of a fiddle for displaying the right components of the quaternion...
    plt.plot(t,head_orientation_q[:,1:4])
    plt.title("Vector components of the quaternions vs time")
    plt.xlabel("Time [s]")
    plt.show()

    # Animation stuff
    create_html()
    create_js()
    # Open the HTML file in the default web browser
    webbrowser.open('Exercise_2\\Outputs\\index.html')

# Animation functions
def create_html():
    # Define the content of the HTML file
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Three.js Arrow Animation</title>
        <style>
        body {{
            margin: 0;
            overflow: hidden;
        }}
        </style>
    </head>
    <body>
        <script src="https://threejs.org/build/three.min.js"></script>
        <script src="Exercise_2/Outputs/animate.js"></script>
    </body>
    </html>
    """

    # Write the HTML file to disk
    with open('Exercise_2\Outputs\index.html', 'w') as f:
        f.write(html_content)

def create_js():
    js_content = r"""
    // Create a scene
    const scene = new THREE.Scene();

    // Create a camera
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 5;

    // Create a renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    // Create an arrow geometry
    const arrowGeometry = new THREE.Geometry();
    arrowGeometry.vertices.push(new THREE.Vector3(0, 0, 0));
    arrowGeometry.vertices.push(new THREE.Vector3(0, 1, 0));
    arrowGeometry.vertices.push(new THREE.Vector3(0.2, 0.8, 0));
    arrowGeometry.vertices.push(new THREE.Vector3(0, 0.7, 0));
    arrowGeometry.vertices.push(new THREE.Vector3(-0.2, 0.8, 0));
    arrowGeometry.vertices.push(new THREE.Vector3(0, 1, 0));
    arrowGeometry.faces.push(new THREE.Face3(0, 1, 2));
    arrowGeometry.faces.push(new THREE.Face3(0, 3, 2));
    arrowGeometry.faces.push(new THREE.Face3(0, 4, 2));
    arrowGeometry.computeBoundingSphere();

    // Create a material for the arrow
    const arrowMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });

    // Create an arrow mesh
    const arrowMesh = new THREE.Mesh(arrowGeometry, arrowMaterial);
    scene.add(arrowMesh);

    // Animate the arrow
    function animate() {
    requestAnimationFrame(animate);
    arrowMesh.rotation.z += 0.1;
    renderer.render(scene, camera);
    }

    animate();
    """

    with open('Exercise_2\\Outputs\\animate.js', 'w') as f:
        f.write(js_content)




if __name__ == '__main__':
    main()





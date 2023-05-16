
"""
Exercise 2: Simulation of a Vestibular Implant
Authors: Quillan Favey, Alessandro Pasini
Version: 1.1
Date: 25.04.2023

This file contains code for simulating the output of a 
vestibular implant using data from an IMU. It takes a .txt input
file containing data from an IMU and outputs the following files:

OUTPUTS: 
      FILES                     DESCRIPTION                                                         VALUES
    - CupularDisplacement.txt   MIN and MAX cupular deflections in [mm]                             [-0.12843, 0.12035]
    - MaxAcceleration.txt       MIN and MAX accelleration sensed by otholith hair cell in [m/s2]    [-5.62990, 6.87026]
    - Nose_final.txt            Final nose orientation (x,y,z)                                      [0.99691, 0.05903, 0.05185]
    - index.html (temp)         To visualize the head orientation during the simulation
    - Main.js (temp)            To visualize the head orientation during the simulation


NAMING CONVENTIONS: 

    - a_:                       an acceleration vector
    -  R:                       rotation matrix
    - _q:                       quaternion
    - _v:                       vector
    - HC:                       head coordinates 
    - IC:                       IMU coordinates

NOTES:
    - Make sure you are connected to the internet or the animation will not work
    - ...
TODO:
MISC
- adjusted means relative to the head <done>
- write as functions <done>
- loading wheels for each function <done> 
- 3D animation <done>
- Check the names of the output files and if the output is correct
- Make it that the paths can run on every system (not just our git file structure)
- Propre

OPTIONAL
- Add a thread for the live server <done>
- Add simulation of cupular deflections in the animation

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
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler




# Main function

def main():
    """
    The main function
    """
    
    # 1. Read in the data 

    # Read data from IMU
    in_file = 'Exercise_2\MovementData\Walking_02.txt'
    acc, omega, rate, n_samples = get_data_sensor(in_file)

    # 2. Find q0, i.e. the orientation of the IMU at t=0 re space

    # Set initial accelerations sensed by the IMU
    a_IMU_base_IC = np.r_[0, 0, -9.81]
    a_IMU_head_IC = np.r_[0, 9.81, 0]
    a_IMU_t0 = acc[0, :]

    # get adjusted accelerations and omegas 
    acc_adjusted, omegas_adjusted, q_0 = align_sensor_vectors(
        a_IMU_base_IC, a_IMU_head_IC, a_IMU_t0, acc, omega)
   
    # 3. Find the orientation of the right SCC (semicircular canal) at t=0

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

    # Normalize the vectors:
    for side in ['right', 'left']:
        Canals[side] = (Canals[side].T /
                        np.sqrt(np.sum(Canals[side]**2, axis=1))).T

    # now we can adjust n0 to the head coordinates
    R_rot = rotmat('y', -15)
    Right_horizontal_SCC = Canals['right'][0]
    n0_HC = R_rot.dot(Right_horizontal_SCC.T).T

    # 4. Using q0, n0 and ω(t) sensor , we calculate stim, the stimulation of the right SCC
    stim = omegas_adjusted @ n0_HC

    # 5. Next we find the canal-transfer-function re velocity.

    T1 = 0.01  # Define time constants [s]
    T2 = 5
    num = [T1*T2, 0]  # Define numerator
    den = [T1*T2, T1+T2, 1]  # Define denominator
    t = np.arange(0, n_samples)/rate  # Define time array (from 50 Hz sample rate)
    cupula_defl = get_cupular_deflections(num, den, t, stim)    # simulate and get cupular deflections

    # convert to mm
    r_canal = 3.2
    cupula_defl = cupula_defl*r_canal
    np.savetxt("Exercise_2\Outputs\CupularDisplacement.txt", [
            np.min(cupula_defl), np.max(cupula_defl)],fmt='%10.5f')   # save to .txt file


    # 7. Using q0 and ω(t) of the sensor , we calculate q(t), i.e. the head orientation re space during the movement
    head_orientation_v, head_orientation_q = calculate_head_orientation(omegas_adjusted)
    np.savetxt("Exercise_2\\Outputs\\Nose_final.txt",head_orientation_v[-1][:],fmt='%10.5f')    # save to .txt file

    # 8. Calculate the stimulation of the otolith hair cell
    stim_otolith = stim_otolith_left(acc_adjusted)
    np.savetxt("Exercise_2\Outputs\MaxAcceleration.txt", [
            np.min(stim_otolith), np.max(stim_otolith)],fmt='%10.5f')   # save to .txt file


    # 9. Show the head orientation, expressed with quaternions as a function of time
    show_head_orientation(head_orientation_q,head_orientation_v,t)

    print('Done!')





# Functions

def running_decorator(func):
    """
    A wrapper function to add a loading wheel to the wrapped function 
    (I thought it would look cool but it is useless...)
    """
    def wrapper(*args, **kwargs):
        msg = f"Running {func.__name__:<{30}}"
        #print(msg, end="\t\t\t\t\t")
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
                print(msg+f"Done.")
                break
            print(msg+char, end="\r")
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


def show_head_orientation(head_orientation_q,head_orientation_v,t,path="Exercise_2\\Outputs\\"):
    """
    Creates a .avi file playing the nose orientation as a function of time or maybe 
    a .js file that will be opened in the browser
    ----------
    INPUTS:
    head_orientation_q: Head orientation quaternions
    head_orientation_v: Head orientation vectors
    t:                  time
    path:               Path in which we generate the output files
    ----------
    Returns: Void
    """
    print("Showing head orientation @: http:\\127.0.0.1:8000\\Exercise_2\\Outputs\\index.html")
    # A bit of a fiddle for displaying the right components of the quaternion...
    plt.plot(t,head_orientation_q[:,1:4])
    plt.title("Vector components of the quaternions vs time")
    plt.xlabel("Time [s]")
    #plt.show()

    # Animation stuff

    # convert python array to js array
    js_array = f""
    for i in head_orientation_v:
        pos = f""
        for j in i:
            pos=pos+f"{j},"
        js_array = js_array + f"[{pos[0:-1]}], "
    js_array = js_array[0:-2]

    # create the files for the animation
    create_html(path)
    create_js(js_array,path)

    # Start live server and Open the HTML file in the default web browser
    server = MyServer()
    webbrowser.open('http://127.0.0.1:8000/Exercise_2/Outputs/index.html')
    server.start()

    # Handle when the user wants to exit
    stop = ""
    while stop != "q":
        stop = input("press q to quit:\n")
    server.stop()    
    
    # cleanup files
    os.remove(path+'Main.js')
    os.remove(path+'index.html')


# Server classes for the serving the animation

class NoLogRequestHandler(SimpleHTTPRequestHandler):
    """
    A handler so that there is no logging msg in the console
    """
    def log_message(self, format, *args):
        pass


class MyServer(threading.Thread):
    """
    A simple threaded server to handle requests for displaying the animation
    """
    def run(self):
        self.server = ThreadingHTTPServer(('localhost', 8000), NoLogRequestHandler)
        self.server.serve_forever()
    def stop(self):
        self.server.shutdown()



# Functions to create the html and js files for the animation

def create_html(path):
    """
    Generate a html file to display the animation
    ----------
    path:   Path to which we generate the file
    ----------
    Returns: Writes a .html file in Exercise_2\\Outputs\\index.html
    """

    # Define the content of the HTML file
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Viewer</title>
        <style>
        body {{
            margin: 0;
            overflow: hidden;
        }}
        #info {{
	position: absolute;
	top: 10px;
	width: 100%;
	text-align: center;
	z-index: 100;
	display:block;
}}
h1,p,button{{
    font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif
}}


        </style>
    </head>
    <body>
        <script async src="https://unpkg.com/es-module-shims@1.6.3/dist/es-module-shims.js"></script> <script type="importmap"> {{ "imports": {{ "three": "https://unpkg.com/three@v0.149.0/build/three.module.js", "three/addons/": "https://unpkg.com/three@v0.149.0/examples/jsm/" }} }} </script>
        <script type="module" src="Main.js"></script>
        <div id="info">
            <h1>Nose orientation during infinity walk</h1>
            <p>Click and drag to rotate</p>
            <p><span style="color:#ff0000">X-axis</span>, <span style="color:#00ff00">Y-axis</span>, <span style="color:#0000ff">Z-axis</span>,<span style="color:#00ffff"> Nose</span></p>
            <button onClick="window.location.reload();">Replay</button>
        </div>

    </body>
    </html>
    
    
    
    """

    # Write the HTML file to disk
    with open(path+'index.html', 'w') as f:
        f.write(html_content)

def create_js(js_array,path):
    """
    Generate a js file that displays a coordinate system, a vector and a head in 3D in the /Outputs folder
    ----------
    INPUTS:
    js_array:   An array containing 3D coordinates
    ----------
    Returns: Writes a .js file in Exercise_2\\Outputs\\Main.js
    """
    js_content = f"""
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';
import {{ DRACOLoader }} from 'three/addons/loaders/DRACOLoader.js';

// Create a renderer
var renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;

// Add the renderer to the HTML document
document.body.appendChild(renderer.domElement);

// Create a camera
var camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
const controls = new OrbitControls(camera, renderer.domElement);
camera.position.z = 5;
camera.position.y = 3;
camera.position.x = 1
controls.target = new THREE.Vector3(0, 0, 0)
controls.update();

// create material
const format = ( renderer.capabilities.isWebGL2 ) ? THREE.RedFormat : THREE.LuminanceFormat;


  const colors = new Uint8Array( 0 + 2 );

  for ( let c = 0; c <= colors.length; c ++ ) {{

    colors[ c ] = ( c / colors.length ) * 256;

  }}

  const gradientMap = new THREE.DataTexture( colors, colors.length, 1, format );
  gradientMap.needsUpdate = true;

const PinkMaterial = new THREE.MeshToonMaterial({{
  color: 'pink',
}});


// loader


    


// Create a scene
var scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff)



// a light is required for MeshPhongMaterial to be seen
const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8)
directionalLight.position.z = 3
directionalLight.position.x = 1
directionalLight.position.y = 3
scene.add(directionalLight)


// Create two arrows with different colors
var x = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 0), 1, 0xff0000);
var y = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 0), 1, 0x00ff00);
var z = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 0), 1, 0x0000ff);

// Add the arrows to the scene
scene.add(x);
scene.add(y);
scene.add(z);


// Render the scene
renderer.render(scene, camera);

//animate the scene
const fps = 50;
let positions = [{js_array}]

let i = 0;

const gltfLoader = new GLTFLoader();
gltfLoader.load('scene.gltf', function(gltf) {{
  gltf.scene.traverse(function(child) {{
    if (child instanceof THREE.Mesh) {{
      child.material = PinkMaterial;
    }}
  }});
  const object = gltf.scene;
  object.position.set(0, -0.19, 0);
  scene.add(gltf.scene);



function animate() {{
    var nose = new THREE.ArrowHelper(new THREE.Vector3(positions[i][0],positions[i][2],positions[i][1]), new THREE.Vector3(0, 0, 0), 1, 0x00ffff);
    scene.add(nose)
    object.lookAt(new THREE.Vector3(positions[i][0],positions[i][2]-0.2,positions[i][1]));
    
    renderer.render(scene, camera);
    i = i + 1;
    setTimeout(() => {{
        requestAnimationFrame(animate);

    }}, 1000 / fps);
    scene.remove(nose)
}}
animate();
    
}});      
    
    """

    with open(path+'Main.js', 'w') as f:
        f.write(js_content)


if __name__ == '__main__':
    main()






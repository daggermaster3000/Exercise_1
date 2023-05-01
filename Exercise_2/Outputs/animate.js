import { OrbitControls } from "./vendor/three/examples/jsm/controls/OrbitControls.js";
// Create a renderer
var renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);

// Add the renderer to the HTML document
document.body.appendChild(renderer.domElement);

// Create a camera
var camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
controls = new OrbitControls( camera, renderer.domElement );
camera.position.z = 5;
controls.target.set(0,0,0);
controls.update();

// Create a scene
var scene = new THREE.Scene();

// Create two arrows with different colors
var x = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 0), 1, 0xff0000);
var y = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 0), 1, 0x00ff00);
var z = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 0), 1, 0x0000ff);

// Add the arrows to the scene
scene.add(x);
scene.add(y);
scene.add(z)

// Render the scene
renderer.render(scene, camera);

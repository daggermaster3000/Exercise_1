# Import the required packages
import skinematics as skin
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# We calculate the quaternion to go from g_upright to g_rotated (q_adjust)
# by going back to rotation matrix form (r_adjust) we can rotate from upright to adjust
# Enter the measurements
g_upright = [9.81, 0, 0]
g_rotated = [9.75, 0.98, -0.39]
# Calculate the sensor orientation
q_adjust = skin.vector.q_shortest_rotation(g_upright, g_rotated)
q_upright = [0, np.sin(np.pi/4), 0]
q_total = skin.quat.Quaternion(skin.quat.q_mult(q_upright, q_adjust))

# Convert to matrices
r_total = q_total.export('rotmat')
q_adjust = skin.quat.Quaternion(skin.vector.q_shortest_rotation(g_upright, g_rotated))
r_adjust = q_adjust.export('rotmat')
#print( r_adjust @ g_upright)

# Alternatively we can also use skin.vector.rotate_vector(g_upright, q_adjust)
start = np.r_[0,0,9.81]
end = np.r_[0,-9.81,0]
q_short = skin.vector.q_shortest_rotation(start,end)
r_short = ...
end2 = skin.vector.rotate_vector(start, q_short)


#plot

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(0, 0, 0, 0, 0, 9.81)
ax.quiver(0,0,0,0,-9.81,0)
ax.scatter(end2[0],end2[1],end2[2])

ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([-10, 10])
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()
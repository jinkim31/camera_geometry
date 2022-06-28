import numpy as np
import cv2
import matplotlib.pyplot as plt

IMAGE_RESOLUTION = [20, 20]

# given projection matrix
extrinsic = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, -2],
                      [0, 0, 0, 1]])

intrinsic = np.array([[1, 0, IMAGE_RESOLUTION[0]/2, 0],
                      [0, 1, IMAGE_RESOLUTION[1]/2, 0],
                      [0, 0, 1, 0]])

# generate 3d world points
# points_world = 20 * np.random.sample((10, 3)) - 10
points_world = np.array([[0, 0, 0],
                         [0, 0, 0],
                         [10, -10, 0],
                         [10, 0, 0],
                         [10, 10, 0],
                         [0, 10, 0],
                         [-10, 10, 0],
                         [-10, 0, 0],
                         [-10, -10, 0],
                         [0, -10, 0]])
points_world = np.concatenate((points_world, np.ones((10, 1))), axis=1)

# get projected homogeneous coordinate
points_image = np.linalg.multi_dot([intrinsic, extrinsic, points_world.transpose()])

# normalize
points_image[0] /= points_image[2]
points_image[1] /= points_image[2]

# draw figures
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.view_init(azim=-90, elev=90)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim3d(-10, 10)
ax.set_ylim3d(-10, 10)
ax.set_zlim3d(-10, 10)
ax.scatter3D(points_world[:, 0], points_world[:, 1], points_world[:, 2])
ax = fig.add_subplot(1, 2, 2)
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)
ax.scatter(points_image[0], points_image[1])
plt.show()

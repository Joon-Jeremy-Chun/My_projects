# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:05:17 2024

@author: joonc
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Define atom positions for a tetrahedral molecule (CH4 as an example)
atoms = np.array([
    [0, 0, 0],  # Central atom
    [1, 1, 1],
    [1, -1, -1],
    [-1, 1, -1],
    [-1, -1, 1]
])

# Symmetry operation: 90-degree rotation around the z-axis
def rotate_z(points, angle):
    angle = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    return np.dot(points, rotation_matrix.T)

# Original and rotated positions
rotated_atoms = rotate_z(atoms, 90)

# Plot the molecule and its rotated state
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(atoms[:, 0], atoms[:, 1], atoms[:, 2], c='b', label='Original', s=100)
ax.scatter(rotated_atoms[:, 0], rotated_atoms[:, 1], rotated_atoms[:, 2], c='r', label='Rotated', s=100)

# Label each point with a number
for i, (x, y, z) in enumerate(atoms):
    ax.text(x, y, z, f'{i}', color='blue')
for i, (x, y, z) in enumerate(rotated_atoms):
    ax.text(x, y, z, f'{i}', color='red')

# Label axes and show legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()


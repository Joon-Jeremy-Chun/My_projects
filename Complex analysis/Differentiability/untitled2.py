# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:44:52 2024

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

# Adjust the size of the points using the 's' parameter
ax.scatter(atoms[:, 0], atoms[:, 1], atoms[:, 2], c='b', label='Original', s=100)  # Original points (size 100)
ax.scatter(rotated_atoms[:, 0], rotated_atoms[:, 1], rotated_atoms[:, 2], c='r', label='Rotated', s=100)  # Rotated points (size 100)

# Add connections (chemical bonds) for the original molecule
for i in range(1, len(atoms)):  # Skip the central atom (index 0)
    ax.plot(
        [atoms[0, 0], atoms[i, 0]],  # X coordinates
        [atoms[0, 1], atoms[i, 1]],  # Y coordinates
        [atoms[0, 2], atoms[i, 2]],  # Z coordinates
        color='blue'
    )

# Add connections (chemical bonds) for the rotated molecule
for i in range(1, len(rotated_atoms)):  # Skip the central atom (index 0)
    ax.plot(
        [rotated_atoms[0, 0], rotated_atoms[i, 0]],  # X coordinates
        [rotated_atoms[0, 1], rotated_atoms[i, 1]],  # Y coordinates
        [rotated_atoms[0, 2], rotated_atoms[i, 2]],  # Z coordinates
        color='red'
    )

# Label each point with a number
for i, (x, y, z) in enumerate(atoms):
    ax.text(x, y, z, f'{i}', color='blue')
for i, (x, y, z) in enumerate(rotated_atoms):
    ax.text(x, y, z, f'{i}', color='red')

# Set equal aspect ratio
max_range = np.array([atoms[:, 0].max() - atoms[:, 0].min(),
                      atoms[:, 1].max() - atoms[:, 1].min(),
                      atoms[:, 2].max() - atoms[:, 2].min()]).max() / 2.0

mid_x = (atoms[:, 0].max() + atoms[:, 0].min()) * 0.5
mid_y = (atoms[:, 1].max() + atoms[:, 1].min()) * 0.5
mid_z = (atoms[:, 2].max() + atoms[:, 2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Label axes and show legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()

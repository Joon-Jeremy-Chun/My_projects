# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:47:14 2024

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

# Plot atoms and connections
ax.scatter(atoms[:, 0], atoms[:, 1], atoms[:, 2], c='b', label='Original', s=100)
ax.scatter(rotated_atoms[:, 0], rotated_atoms[:, 1], rotated_atoms[:, 2], c='r', label='Rotated', s=100)

# Add connections for the original molecule
for i in range(1, len(atoms)):
    ax.plot(
        [atoms[0, 0], atoms[i, 0]], 
        [atoms[0, 1], atoms[i, 1]], 
        [atoms[0, 2], atoms[i, 2]], 
        color='blue'
    )

# Add connections for the rotated molecule
for i in range(1, len(rotated_atoms)):
    ax.plot(
        [rotated_atoms[0, 0], rotated_atoms[i, 0]], 
        [rotated_atoms[0, 1], rotated_atoms[i, 1]], 
        [rotated_atoms[0, 2], rotated_atoms[i, 2]], 
        color='red'
    )

# Label points
for i, (x, y, z) in enumerate(atoms):
    ax.text(x, y, z, f'{i}', color='blue')
for i, (x, y, z) in enumerate(rotated_atoms):
    ax.text(x, y, z, f'{i}', color='red')

# Set equal aspect ratio for all axes
def set_axes_equal(ax):
    '''Make the axes of a 3D plot have equal scale so the plot is a cube.'''
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    spans = abs(limits[:, 1] - limits[:, 0])
    max_span = max(spans)
    centers = np.mean(limits, axis=1)
    bounds = np.array([centers - max_span / 2, centers + max_span / 2]).T
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_zlim(bounds[2])

# Apply equal aspect ratio
set_axes_equal(ax)

# Label axes and show legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()

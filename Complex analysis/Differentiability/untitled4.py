# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:49:03 2024

@author: joonc
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Positions for methane (CH4): carbon at the center, hydrogens in a tetrahedral shape
carbon = np.array([0, 0, 0])  # Central atom
hydrogens = np.array([
    [1, 1, 1],   # Hydrogen 1
    [1, -1, -1], # Hydrogen 2
    [-1, 1, -1], # Hydrogen 3
    [-1, -1, 1]  # Hydrogen 4
])

# Combine the positions into one array for easy plotting
atoms = np.vstack((carbon, hydrogens))

# Plot the molecule
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the central carbon atom
ax.scatter(carbon[0], carbon[1], carbon[2], color='black', s=300, label='Carbon (C)')

# Plot the hydrogen atoms
ax.scatter(hydrogens[:, 0], hydrogens[:, 1], hydrogens[:, 2], color='blue', s=200, label='Hydrogen (H)')

# Draw bonds between the carbon and hydrogens
for hydrogen in hydrogens:
    ax.plot(
        [carbon[0], hydrogen[0]],
        [carbon[1], hydrogen[1]],
        [carbon[2], hydrogen[2]],
        color='gray'
    )

# Label the hydrogen atoms
for i, (x, y, z) in enumerate(hydrogens, start=1):
    ax.text(x, y, z, f'H{i}', color='blue')

# Label the central carbon atom
ax.text(carbon[0], carbon[1], carbon[2], 'C', color='black')

# Set equal aspect ratio for proper visualization
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
plt.title("Methane (CH₄) Molecule")
plt.show()

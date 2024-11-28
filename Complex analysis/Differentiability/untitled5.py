# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:51:12 2024

@author: joonc
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Define positions for carbons and hydrogens in allene (C₃H₄)
carbons = np.array([
    [0, 0, 0],   # Central carbon (C1)
    [-1.5, 0, 0],  # Left carbon (C2)
    [1.5, 0, 0]   # Right carbon (C3)
])

hydrogens = np.array([
    [-1.5, 1, 1],   # H1 bonded to C2
    [-1.5, -1, -1], # H2 bonded to C2
    [1.5, 1, -1],   # H3 bonded to C3
    [1.5, -1, 1]    # H4 bonded to C3
])

# Bonds: Define connections as pairs of indices in the 'carbons' and 'hydrogens' arrays
bonds = [
    (0, 1),  # Double bond between C1 and C2
    (0, 2),  # Double bond between C1 and C3
    (1, 3),  # Bond between C2 and H1
    (1, 4),  # Bond between C2 and H2
    (2, 5),  # Bond between C3 and H3
    (2, 6)   # Bond between C3 and H4
]

# Combine all atoms for plotting
all_atoms = np.vstack((carbons, hydrogens))

# Plot the molecule
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot carbon atoms
ax.scatter(carbons[:, 0], carbons[:, 1], carbons[:, 2], c='black', s=300, label='Carbon (C)')

# Plot hydrogen atoms
ax.scatter(hydrogens[:, 0], hydrogens[:, 1], hydrogens[:, 2], c='blue', s=200, label='Hydrogen (H)')

# Draw bonds
for bond in bonds:
    atom1 = all_atoms[bond[0]]
    atom2 = all_atoms[bond[1]]
    ax.plot(
        [atom1[0], atom2[0]],
        [atom1[1], atom2[1]],
        [atom1[2], atom2[2]],
        color='gray',
        linewidth=2
    )

# Label each atom
for i, (x, y, z) in enumerate(carbons):
    ax.text(x, y, z, f'C{i+1}', color='black')
for i, (x, y, z) in enumerate(hydrogens):
    ax.text(x, y, z, f'H{i+1}', color='blue')

# Set equal aspect ratio
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

set_axes_equal(ax)

# Label axes and show legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title("Allene (C₃H₄) Molecule")
plt.show()

# -*- coding: utf-8 -*-
"""
Created on Wed May  1 17:10:54 2024

@author: joonc
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the number of steps in the random walk
n_steps = 10

# Define the step options for a 2D random walk and their probabilities
# Right, Left, Up, Down
# step_options = np.array([[1, 0], [1, 1], [-1, 0], [-1, 1], [0, 1], [-1, -1], [0, -1], [1, -1]])
# probabilities = np.array([1/8,   1/8,    1/8,     1/8,     1/8,    1/8,      1/8,     1/8])  # Equal probability for each direction

step_options = np.array([[1, 0], [1, 1], [-1, 0], [-1, 1], [0, 1], [-1, -1], [0, -1], [1, -1]])
probabilities = np.array([2/64, 2/64,   0/64,   0/64,    29/64,   0/64,     29/64,     2/64])  # Biased probability 

# Total number of simulations
n_simulations = 10000

# Perform the Monte Carlo simulations
positions = np.zeros((n_simulations, 2))
for i in range(n_simulations):
    steps = np.random.choice(len(step_options), size=n_steps, p=probabilities)
    walk = np.sum(step_options[steps], axis=0)
    positions[i, :] = walk

# Prepare data for 3D histogram: X, Y coordinates
x = positions[:, 0]
y = positions[:, 1]

# Create histogram bins
bin_size = 3
x_bins = np.arange(start=min(x), stop=max(x) + bin_size, step=bin_size)
y_bins = np.arange(start=min(y), stop=max(y) + bin_size, step=bin_size)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Compute histogram
hist, xedges, yedges = np.histogram2d(x, y, bins=(x_bins, y_bins))

# Construct arrays for the anchor positions of the bars.
xpos, ypos = np.meshgrid(xedges[:-1] + bin_size/2, yedges[:-1] + bin_size/2, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the 16 bars.
dx = dy = np.ones_like(zpos) * bin_size
dz = hist.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Frequency')

plt.title('3D Histogram of 2D Random Walk Final Positions')
plt.show()

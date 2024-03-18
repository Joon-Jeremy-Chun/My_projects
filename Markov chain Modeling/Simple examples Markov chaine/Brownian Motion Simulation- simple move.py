# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 16:01:29 2024

@author: joonc
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the number of steps
n_steps = 1000

# Define the initial position
initial_position = np.array([0, 0])

# Define the step options (up, down, left, right)
step_options = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

# Initialize the position array
positions = np.zeros((n_steps + 1, 2))
positions[0] = initial_position

# Simulate the Brownian motion
for i in range(1, n_steps + 1):
    # Choose a random step from the step options
    random_step = step_options[np.random.randint(4)]
    # Update the position
    positions[i] = positions[i - 1] + random_step

# Plot the trajectory
plt.plot(positions[:, 0], positions[:, 1])
plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("Brownian Motion Simulation")
plt.grid(True)
plt.show()
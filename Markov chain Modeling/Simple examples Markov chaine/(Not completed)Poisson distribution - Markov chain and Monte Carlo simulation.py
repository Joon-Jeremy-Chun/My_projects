# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 19:31:39 2024

@author: joonc
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the number of steps
n_steps = 500

# Define the initial position
initial_position = np.array([0.0, 0.0], dtype=np.float64)

# Define the step options (right, not move, left)
step_options = np.array([[1, 0], [0, 0], [-1, 0]])

# Initialize the position array
positions = np.zeros((n_steps + 1, 2))
positions[0] = initial_position

# # Simulate the Brownian motion
# for i in range(1, n_steps + 1):
#     if positions[i - 1][0] == 0:
#         # If the x-coordinate is 0, not move left
#         random_step = step_options[0]
#     else:
#         # Otherwise, choose a random step
#         random_step = step_options[np.random.randint(3)]
#     positions[i] = positions[i - 1] + random_step

# # Plot the trajectory
# plt.plot(positions[:, 0], positions[:, 1])
# plt.xlabel("X position")
# plt.ylabel("Y position")
# plt.title("Brownian Motion Simulation")
# plt.grid(True)
# plt.show()

# Monte Carlo simulation for final x-coordinates
n_simulations = 10000
final_x_coordinates = np.zeros(n_simulations)

for i in range(n_simulations):
    position = initial_position.copy()
    for _ in range(n_steps):
        if position[0] == 0:
            # If the x-coordinate is 0, not move left
            random_step = step_options[np.random.randint(2)]
        else:
            # Otherwise, choose a random step
            random_step = step_options[np.random.randint(3)]
        position += random_step
    final_x_coordinates[i] = position[0]

# Create a histogram for final x-coordinates
plt.hist(final_x_coordinates, bins=20, edgecolor='black')
plt.xlabel("Final X-Coordinate")
plt.ylabel("Frequency")
plt.title("Distribution of Final X-Coordinates from Monte Carlo Simulations")
plt.grid(True)
plt.show()
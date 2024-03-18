# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 16:12:06 2024

@author: joonc
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the number of steps
n_steps = 20

# Define the initial position
initial_position = np.array([0.0, 0.0], dtype=np.float64)

# Define the step options (up, down, left, right)
step_options = np.array([[1, 0], [-1, 0]])

# Initialize the position array
positions = np.zeros((n_steps + 1, 2))
positions[0] = initial_position

# Simulate the Brownian motion
for i in range(1, n_steps + 1):
    random_step = step_options[np.random.randint(2)]  # Choose a random step
    positions[i] = positions[i - 1] + random_step

# Plot the trajectory
plt.plot(positions[:, 0], positions[:, 1])
plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("Brownian Motion Simulation")
plt.grid(True)
plt.show()

# Monte Carlo simulation for final x-coordinates
n_simulations = 100000
final_x_coordinates = np.zeros(n_simulations)

for i in range(n_simulations):
    position = np.array([0.0, 0.0], dtype=np.float64)
    for _ in range(n_steps):
        random_step = step_options[np.random.randint(2)]
        position += random_step
    final_x_coordinates[i] = position[0]

# Now `final_y_coordinates` contains the final y-coordinates from 100,000 simulations
# You can analyze or visualize this data as needed

print (final_x_coordinates)

# Create a bar graph for final x-coordinates
plt.hist(final_x_coordinates, bins=20, edgecolor='black')
plt.xlabel("Final X-Coordinate")
plt.ylabel("Frequency")
plt.title("Distribution of Final X-Coordinates from Monte Carlo Simulations")
plt.grid(True)
plt.show()

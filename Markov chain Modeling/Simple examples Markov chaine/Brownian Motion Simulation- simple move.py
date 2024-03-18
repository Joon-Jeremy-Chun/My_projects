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
initial_position = np.array([0.0, 0.0], dtype=np.float64)

# Define the step options (up, down, left, right)
#step_options = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
step_options = np.array([[1, 0], [-1, 0], [0, -1]])
#step_options = np.array([[1, 0], [1, 1], [-1, 0] [-1, 1], [0, 1], [-1, -1], [0, -1] [1, -1]])
# Initialize the position array
positions = np.zeros((n_steps + 1, 2))
positions[0] = initial_position

# Simulate the Brownian motion
for i in range(1, n_steps + 1):
    # Choose a random step from the step options
    #random_step = step_options[np.random.randint(4)]
    random_step = step_options[np.random.randint(3)]
    #random_step = step_options[np.random.randint(8)]
    # Update the position
    positions[i] = positions[i - 1] + random_step

# Plot the trajectory
plt.plot(positions[:, 0], positions[:, 1])
plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("Brownian Motion Simulation")
plt.grid(True)
plt.show()

final_position = positions[-1]
print(f"The final position in coordinates is X: {final_position[0]:.2f}, Y: {final_position[1]:.2f}")
#%%
#mport numpy as np

# Set the number of simulations
n_simulations = 100000

# Initialize an array to store final y-coordinates
final_y_coordinates = np.zeros(n_simulations)

# Simulate Brownian motion and record final y-coordinates
for i in range(n_simulations):
    position = np.array([0.0, 0.0], dtype=np.float64)
    for _ in range(n_steps):
        random_step = step_size * np.random.randn(2)
        position += random_step
    final_y_coordinates[i] = position[1]

# Now `final_y_coordinates` contains the final y-coordinates from 100,000 simulations
# You can analyze or visualize this data as needed
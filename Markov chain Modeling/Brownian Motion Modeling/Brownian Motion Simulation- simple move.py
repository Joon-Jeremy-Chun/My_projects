# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 16:01:29 2024

@author: joonc
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the number of steps
n_steps = 100

# Define the initial position
initial_position = np.array([0.0, 0.0], dtype=np.float64)

# Define the step options 
#step_options = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
step_options = np.array([[1, 0], [-1, 0]])
#step_options = np.array([[1, 0], [1, 1], [-1, 0], [-1, 1], [0, 1], [-1, -1], [0, -1], [1, -1]])
#step_options = np.array([[1, 0], [np.sqrt(2)/2, np.sqrt(2)/2], [-1, 0], [-np.sqrt(2)/2, np.sqrt(2)/2], [0, 1], [-np.sqrt(2)/2, -np.sqrt(2)/2], [0, -1], [np.sqrt(2)/2, -np.sqrt(2)/2]])
# Initialize the position array
positions = np.zeros((n_steps + 1, 2))
positions[0] = initial_position

# Simulate the Brownian motion
for i in range(1, n_steps + 1):
    # Choose a random step from the step options
    #random_step = step_options[np.random.randint(4)]
    random_step = step_options[np.random.choice([0,1], p = [1/2,1/2])]
    #random_step = step_options[np.random.choice([0,1,2,3,4,5,6,7], p = [1/8,1/8,1/8,1/8,1/8,1/8,1/8,1/8])]
    
    # Update the position
    positions[i] = positions[i - 1] + random_step

# Plot the trajectory
plt.plot(positions[:, 0], positions[:, 1])
plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("Brownian Motion Simulation")
plt.grid(True)

# Highlight the initial and final points
plt.scatter(*initial_position, color='green', s=100, zorder=5, label='Start')
plt.scatter(*positions[-1], color='red', s=100, zorder=5, label='End')

# Set limits for x and y axes
plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.show()

final_position = positions[-1]
print(f"The final position in coordinates is X: {final_position[0]:.2f}, Y: {final_position[1]:.2f}")

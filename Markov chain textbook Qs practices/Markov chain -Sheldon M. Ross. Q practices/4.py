# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 19:15:27 2024

@author: joonc
"""

import numpy as np
import matplotlib.pyplot as plt

def lightning_step(position, step_range):
    """Randomly move the lightning bolt one step."""
    step = step_range * (2 * np.random.random(2) - 1)
    return position + step

def simulate_lightning(start_pos, steps, step_range):
    """Simulate the path of a lightning bolt."""
    path = [start_pos]
    for _ in range(steps):
        current_pos = path[-1]
        new_pos = lightning_step(current_pos, step_range)
        path.append(new_pos)
    return np.array(path)

# Parameters
start_pos = np.array([0, 10])  # Starting point of the lightning
steps = 100  # Number of steps in the simulation
step_range = 0.2  # Maximum distance moved in one step

# Simulate lightning
path = simulate_lightning(start_pos, steps, step_range)

# Plot
plt.figure(figsize=(6, 8))
plt.plot(path[:, 0], path[:, 1], lw=2)
plt.scatter([start_pos[0]], [start_pos[1]], c='red')  # Starting point
plt.title('Simulated Lightning Path')
plt.xlabel('Horizontal Position')
plt.ylabel('Vertical Position')
plt.grid(True)
plt.show()
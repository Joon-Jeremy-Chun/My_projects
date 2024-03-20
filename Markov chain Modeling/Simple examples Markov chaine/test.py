# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:29:43 2024

@author: joonc
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the number of steps
n_steps = 1000

# Define the initial position
initial_position = np.array([0.0, 0.0], dtype=np.float64)

# Define the step options
step_options = np.array([[1, 0], [1, 1], [-1, 0], [-1, 1], [0, 1], [-1, -1], [0, -1], [1, -1]])

# Initialize the position array
positions = np.zeros((n_steps + 1, 2))
positions[0] = initial_position

# Function to update plot in animation
def update(frame):
    if frame == 0:
        return ax.plot(positions[:1, 0], positions[:1, 1], 'bo')  # Plot initial position
    else:
        step = step_options[np.random.randint(8)]
        positions[frame] = positions[frame - 1] + step
        return ax.plot(positions[:frame+1, 0], positions[:frame+1, 1], 'bo-')  # Plot trajectory up to current frame

# Function to initialize the plot
def init():
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_title("Brownian Motion Simulation")
    ax.grid(True)
    return []

# Create the animation
fig, ax = plt.subplots(figsize=(8, 8))
ani = FuncAnimation(fig, update, frames=n_steps+1, init_func=init, blit=True, repeat=True)

# Keep a reference to the animation object until it's rendered
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:06:49 2024

@author: joonc
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the number of steps
n_steps = 100

# Define the initial position
initial_position = np.array([0.0, 0.0], dtype=np.float64)

# Define the step options
step_options = np.array([[1, 0], [-1, 0]])
#step_options = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]])
# Initialize the position array
positions = np.zeros((n_steps + 1, 2))
positions[0] = initial_position

# Function to update plot in animation
def update(frame):
    step = step_options[np.random.randint(2)]
    #step = step_options[np.random.randint(8)]
    positions[frame] = positions[frame - 1] + step

    ax.clear()  # Clear previous frame's content
    ax.plot(positions[:frame+1, 0], positions[:frame+1, 1], 'bo-', markersize=3)  # Smaller movement points
    ax.scatter(*initial_position, color='green', s=100, zorder=5)  # Highlight start point
    if frame == n_steps:
        ax.scatter(*positions[frame], color='red', s=100, zorder=5)  # Highlight end point only at the last frame

    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_title("Brownian Motion Simulation")
    ax.grid(True)
    return ax,

# Function to initialize the plot
def init():
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_title("Brownian Motion Simulation")
    ax.grid(True)
    return []

# Create the animation
fig, ax = plt.subplots(figsize=(8, 8))
ani = FuncAnimation(fig, update, frames=n_steps+1, init_func=init, blit=False, repeat=False)

ani.save('brownian_motion-1D.gif', writer='imagemagick', fps=10) # Save as GIF
#ani.save('brownian_motion-2D.gif', writer='imagemagick', fps=10) # Save as GIF
plt.show()
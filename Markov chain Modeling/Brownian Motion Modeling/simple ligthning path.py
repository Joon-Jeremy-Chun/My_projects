# -*- coding: utf-8 -*-
"""
Created on Wed May  1 22:31:45 2024

@author: joonc
"""

import numpy as np
import matplotlib.pyplot as plt

# Set up the simulation parameters
size = 100  # Size of the grid (100x100)
start_point = (0, size//2)  # Start at the middle of the top row
iterations = 1000  # Number of steps in the simulation

# Create an empty grid
grid = np.zeros((size, size))

# Function to simulate lightning movement
def simulate_lightning(start_point, grid, iterations):
    x, y = start_point
    for _ in range(iterations):
        grid[x, y] = 1  # Mark the current position as part of the lightning path
        if x == size - 1:  # Stop if we reach the bottom of the grid
            break
        # Randomly choose the next step direction
        move = np.random.choice(['down', 'left', 'right'], p=[0.8, 0.1, 0.1])
        if move == 'down' and x < size - 1:
            x += 1
        elif move == 'left' and y > 0:
            y -= 1
        elif move == 'right' and y < size - 1:
            y += 1

    return grid

# Run the simulation
grid = simulate_lightning(start_point, grid, iterations)

# Plotting the result
plt.figure(figsize=(10, 10))
plt.imshow(grid, cmap='Greys', interpolation='nearest')
plt.title("Simulation of Lightning Strike in a 100x100 Grid")
plt.axis('off')
plt.show()

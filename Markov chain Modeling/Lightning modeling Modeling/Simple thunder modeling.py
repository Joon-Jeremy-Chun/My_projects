# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:52:14 2024

@author: joonc
"""

import numpy as np
import matplotlib.pyplot as plt

# Grid size
width = 101  # Width of the grid
height = 51  # Height of the grid

# Start point (in the middle of the top row)
start = (height - 1, width // 2)

# Initialize the grid
grid = np.zeros((height, width))

# Mark the starting point
grid[start] = 1

def lightning_step(grid, current_pos):
    """Perform one step of the lightning bolt."""
    y, x = current_pos
    
    # Potential moves: down, down-left, down-right
    moves = [(y - 1, x), (y - 1, max(0, x - 1)), (y - 1, min(width - 1, x + 1))]
    
    # Choose a move randomly
    next_pos = moves[np.random.randint(0, len(moves))]
    
    # Update the grid
    grid[next_pos] = 1
    
    return next_pos

# Current position of the lightning bolt
current_pos = start

# Simulate the lightning bolt until it reaches the bottom
while current_pos[0] > 0:
    current_pos = lightning_step(grid, current_pos)

# Plot the result
plt.figure(figsize=(10, 20))
plt.imshow(grid, cmap='Blues', interpolation='nearest')
plt.title('Simplified Lightning Path Simulation')
plt.axis('off')
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 09:57:55 2024

@author: joonc
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Define the number of steps
n_steps = 500

# Define the initial position
initial_position = np.array([0.0, 0.0], dtype=np.float64)

# Define the step options (right, not move, left)
step_options = np.array([[1, 0], [0, 0], [-1, 0]])

# Initialize the position array
positions = np.zeros((n_steps + 1, 2))
positions[0] = initial_position

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

# Calculate the mean and standard deviation for the Poisson distribution
lambda_ = np.mean(final_x_coordinates)
std_dev = np.std(final_x_coordinates)

# Generate Poisson distribution data
x_poisson = np.arange(poisson.ppf(0.001, lambda_), poisson.ppf(0.999, lambda_))
y_poisson = poisson.pmf(x_poisson, lambda_)

# Create a histogram for final x-coordinates
plt.hist(final_x_coordinates, bins=100, edgecolor='black', density=True)
plt.xlabel("Final X-Coordinate")
plt.ylabel("Frequency")
plt.title("Distribution of Final X-Coordinates from Monte Carlo Simulations")

# Plot the Poisson distribution line
plt.plot(x_poisson, y_poisson, 'r-', lw=2, label='Poisson distribution')

# Add legend
plt.legend()

# Show the plot with the Poisson distribution line
plt.grid(True)
plt.show()

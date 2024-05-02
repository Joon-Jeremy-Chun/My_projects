# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 16:12:06 2024

@author: joonc
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the number of steps
n_steps = 100


# Define the step options (right, not move ,left)
step_options = np.array([[1, 0], [0, 0], [-1, 0]])


# Monte Carlo simulation for final x-coordinates
n_simulations = 10000
final_x_coordinates = np.zeros(n_simulations)

for i in range(n_simulations):
    position = np.array([0.0, 0.0], dtype=np.float64)
    for _ in range(n_steps):
        random_step = step_options[np.random.choice([0,2], p = [0.5, 0.5])]
        position += random_step
    final_x_coordinates[i] = position[0]
    
# Now `final_x_coordinates` contains the final x-coordinates from 100,000 simulations
# You can analyze or visualize this data as needed

#print (final_x_coordinates)

# Create a bar graph for final x-coordinates
plt.hist(final_x_coordinates, bins=100, edgecolor='black')
plt.xlabel("Final X-Coordinate")
plt.ylabel("Frequency")
plt.title("Distribution of Final X-Coordinates from Monte Carlo Simulations")
plt.grid(True)
plt.show()

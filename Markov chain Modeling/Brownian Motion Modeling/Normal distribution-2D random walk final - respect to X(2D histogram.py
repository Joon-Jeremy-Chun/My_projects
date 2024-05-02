# -*- coding: utf-8 -*-
"""
Created on Wed May  1 17:22:37 2024

@author: joonc
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Importing seaborn for enhanced visualizations

# Define the number of steps in the random walk
#n_steps = 100
n_steps = 32

# Define the step options for a 2D random walk and their probabilities
# step_options = np.array([[1, 0], [1, 1], [-1, 0], [-1, 1], [0, 1], [-1, -1], [0, -1], [1, -1]])
# probabilities = np.array([1/8]*8)  # Equal probability for each direction

# step_options = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
# probabilities = np.array([1/4]*4)

step_options = np.array([[1, 0], [1, 1], [-1, 0], [-1, 1], [0, 1], [-1, -1], [0, -1], [1, -1]])
probabilities = np.array([2/64, 2/64,   0/64,   0/64,    29/64,   0/64,     29/64,     2/64])  # Biased probability 

# Total number of simulations
n_simulations = 10000

# Perform the Monte Carlo simulations
positions = np.zeros((n_simulations, 2))
for i in range(n_simulations):
    steps = np.random.choice(len(step_options), size=n_steps, p=probabilities)
    walk = np.sum(step_options[steps], axis=0)
    positions[i, :] = walk

# Prepare data for x-coordinate distribution
x = positions[:, 0]

# Plotting the distribution of x-coordinates
plt.figure(figsize=(10, 6))
sns.histplot(x, bins=100, color='blue', edgecolor='black')  # Histogram without KDE
plt.xlabel('Final X-Coordinate')
plt.ylabel('Frequency')
plt.title('Distribution of Final X-Coordinates from Monte Carlo Simulations')
plt.grid(True)
plt.show()


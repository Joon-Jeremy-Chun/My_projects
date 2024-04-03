# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 18:23:01 2024

@author: joonc
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
grid_size = 100  # Size of the grid
diffusion_rate = 0.1  # Rate at which the disease diffuses
initial_infected_center = (50, 50)  # Initial outbreak point
initial_infection_radius = 10  # Radius around the outbreak point that is initially infected
time_steps = 1000  # Number of time steps to simulate

# Initialize the population grid
population = np.zeros((grid_size, grid_size))

# Initialize the infection at the center
for i in range(grid_size):
    for j in range(grid_size):
        if (i - initial_infected_center[0])**2 + (j - initial_infected_center[1])**2 <= initial_infection_radius**2:
            population[i, j] = 1

# Function to update the population based on the diffusion equation
def diffuse(population):
    new_population = population.copy()
    for i in range(1, grid_size-1):
        for j in range(1, grid_size-1):
            new_population[i, j] += diffusion_rate * (population[i+1, j] + population[i-1, j] +
                                                      population[i, j+1] + population[i, j-1] -
                                                      4 * population[i, j])
    return new_population

# Simulate over time
fig, ax = plt.subplots()
for t in range(time_steps):
    population = diffuse(population)
    if t % 10 == 0:  # Update the plot every 10 time steps
        ax.clear()
        ax.imshow(population, cmap='hot', interpolation='nearest')
        ax.set_title(f"Time step: {t}")
        plt.pause(0.1)

plt.show()
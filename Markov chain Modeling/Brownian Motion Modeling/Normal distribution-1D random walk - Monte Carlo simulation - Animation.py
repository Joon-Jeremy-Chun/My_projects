# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:25:50 2024

@author: joonc
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import norm

# Define the number of steps in the random walk (updated to 100)
n_steps = 100

# Define the step options (right, not move, left)
step_options = np.array([[1, 0], [0, 0], [-1, 0]])

# Total number of simulations
n_simulations = 10000

# Empty array to store final x-coordinates
final_x_coordinates = np.zeros(n_simulations)

def update(frame):
    position = np.array([0.0, 0.0], dtype=np.float64)
    for _ in range(n_steps):
        random_step = step_options[np.random.choice([0, 1, 2], p=[0.5, 0.0, 0.5])]
        position += random_step
    final_x_coordinates[frame] = position[0]

    ax.clear()
    # Plot histogram
    count, bins, ignored = ax.hist(final_x_coordinates[:frame + 1], bins=30, density=True, color='blue', alpha=0.7, edgecolor='black')
    # Fit a normal distribution and plot
    mean, std_dev = 0, np.sqrt(n_steps * (0.5*1**2 + 0.5*(-1)**2))  # Standard deviation for binomial -> normal approximation
    x = np.linspace(-50, 50, 1000)
    normal_dist = norm.pdf(x, mean, std_dev)
    ax.plot(x, normal_dist, 'r-', lw=2)

    ax.set_xlabel("Final X-Coordinate")
    ax.set_ylabel("Probability Density")
    ax.set_title("Distribution of Final X-Coordinates\nfrom Monte Carlo Simulations")
    ax.set_xlim(-50, 50)
    ax.set_ylim(0, max(count.max(), normal_dist.max())*1.1)  # Adjust the y-axis limit
    ax.text(-49, max(count.max(), normal_dist.max())*1.05, f'Simulations: {frame + 1}', fontsize=10, color='red')

def init():
    ax.set_xlim(-50, 50)
    ax.set_ylim(0, 0.12)
    ax.set_xlabel("Final X-Coordinate")
    ax.set_ylabel("Probability Density")
    ax.set_title("Distribution of Final X-Coordinates\nfrom Monte Carlo Simulations")

fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, frames=n_simulations, init_func=init, repeat=False, interval=20)

# Uncomment to save the animation
#ani.save('random_walk_distribution(0.5,0.5).gif', writer='imagemagick', fps=10)

plt.show()


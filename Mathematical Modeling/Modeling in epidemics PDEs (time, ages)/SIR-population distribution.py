# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 22:55:59 2024

@author: joonc
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
beta = 0.3  # Infection rate
gamma = 0.1  # Recovery rate

# Time and age sensitivity
h = 0.01  # Time step size
k = 1  # Age step size

# Time and age limits
T = 50  # Total time
A = 100  # Maximum age

# Discretization
num_t = int(T / h) + 1  # Number of time steps
num_a = int(A / k) + 1  # Number of age steps

# Initialize populations
S = np.zeros((num_a, num_t))
I = np.zeros((num_a, num_t))
R = np.zeros((num_a, num_t))

# Population function for Susceptible
mean_age = 50
std_dev = 15
age_scale = 1000  # Scale factor for population size
ages = np.arange(0, A + 1, k)
S[:, 0] = age_scale * np.exp(-0.5 * ((ages - mean_age) / std_dev) ** 2) / (std_dev * np.sqrt(2 * np.pi))

# Initial conditions for Infected and Recovered
#I[mean_age, 0] = 1  # Start with one infected at the mean age
I[:, 0] = 1    # Initial infected population
R[:, 0] = 0  # No recovered individuals initially

# Finite difference method to update the populations
for j in range(num_t - 1):  # Time loop
    for i in range(0, num_a):  # Age loop
        S[i, j+1] = S[i-1, j] - beta * S[i, j] * I[i, j] * h
        I[i, j+1] = I[i-1, j] + beta * S[i, j] * I[i, j] * h - gamma * I[i, j] * h
        R[i, j+1] = R[i-1, j] + gamma * I[i, j] * h

# Visualization
plt.figure(figsize=(14, 5))
plt.subplot(131)
plt.imshow(S, extent=[0, T, 0, A], origin='lower', aspect='auto')
plt.title('Susceptibles')
plt.xlabel('Time')
plt.ylabel('Age')
plt.colorbar()

plt.subplot(132)
plt.imshow(I, extent=[0, T, 0, A], origin='lower', aspect='auto')
plt.title('Infected')
plt.xlabel('Time')
plt.ylabel('Age')
plt.colorbar()

plt.subplot(133)
plt.imshow(R, extent=[0, T, 0, A], origin='lower', aspect='auto')
plt.title('Recovered')
plt.xlabel('Time')
plt.ylabel('Age')
plt.colorbar()

plt.tight_layout()
plt.show()

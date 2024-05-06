# -*- coding: utf-8 -*-
"""
Created on Sun May  5 16:44:49 2024

@author: joonc
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
beta = 0.5  # infection rate
gamma = 0.1  # recovery rate
k = 0.01  # time step size
h = 1  # space/age step size
total_time = 10  # total time to simulate
total_age = 100  # total age range to simulate
M = int(total_age / h)  # number of age steps
N = int(total_time / k)  # number of time steps

# Initial conditions
S = np.zeros((M, N))
I = np.zeros((M, N))
R = np.zeros((M, N))
S[:, 0] = 999  # Assuming all ages start susceptible except the youngest
I[0, 0] = 1    # Initial infected population at youngest age
R[:, 0] = 0    # Initial recovered population

# Simulation with explicit finite differences
for j in range(1, N):
    for i in range(1, M):
        S[i, j] = S[i, j-1] - k * beta * S[i, j-1] * I[i, j-1] + (h-k)/h * S[i-1, j-1] + k/h * S[i-1, j-1]
        I[i, j] = I[i, j-1] + k * beta * S[i, j-1] * I[i, j-1] - k * gamma * I[i, j-1] + (h-k)/h * I[i-1, j-1] + k/h * I[i-1, j-1]
        R[i, j] = R[i, j-1] + k * gamma * I[i, j-1] + (h-k)/h * R[i-1, j-1] + k/h * R[i-1, j-1]

# Plot the results at the final time step
plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, total_age, M), S[:, -1], label='Susceptible')
plt.plot(np.linspace(0, total_age, M), I[:, -1], label='Infected')
plt.plot(np.linspace(0, total_age, M), R[:, -1], label='Recovered')
plt.title('SIR Model Simulation over Age at Final Time')
plt.xlabel('Age')
plt.ylabel('Population')
plt.legend()
plt.grid(True)
plt.show()

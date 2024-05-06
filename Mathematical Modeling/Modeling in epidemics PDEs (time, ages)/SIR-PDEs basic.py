# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 22:02:25 2024

@author: joonc
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
beta = 0.3  # Infection rate
gamma = 0.1  # Recovery rate

#Be aware of the Stability
h = 1  # Time step size
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

# Initial conditions
S[:, 0] = 1  # Initial susceptible population

# I[:, 0] = 0.0001 #Initial infected population (initiated all age groups)


I[20, 0] = 0.0001    # Initial infected population
R[:, 0] = 0     # Initial recovered population

# Finite difference method to update the populations
for j in range(num_t - 1):  # Time loop
    for i in range(0, num_a):  # Age loop
        S[i, j+1] = S[i-1, j] - beta*S[i, j]*I[i, j]*h
        I[i, j+1] = I[i-1, j] + beta*S[i, j]*I[i, j]*h - gamma*I[i, j]*h 
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

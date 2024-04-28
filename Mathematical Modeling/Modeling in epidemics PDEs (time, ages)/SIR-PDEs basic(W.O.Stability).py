# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 10:07:36 2024

@author: joonc
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
beta = 0.3  # Infection rate
gamma = 0.1  # Recovery rate

#Be aware of the Stability
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

# Initial conditions
S[:, 0] = 1000  # Initial susceptible population
I[:, 0] = 1    # Initial infected population
R[:, 0] = 0     # Initial recovered population

# Finite difference method to update the populations
for j in range(num_t - 1):  # Time loop
    for i in range(0, num_a):  # Age loop
        S[i, j+1] = S[i-1, j]*k/h - beta*S[i, j]*I[i, j]*k + (h-k)/h*S[i,j]
        I[i, j+1] = I[i-1, j]*k/h + beta*S[i, j]*I[i, j]*k - gamma*I[i, j]*k+ (h-k)/h*I[i,j] 
        R[i, j+1] = R[i-1, j]*k/h + gamma * I[i, j]*k + (h-k)/h*R[i,j]

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

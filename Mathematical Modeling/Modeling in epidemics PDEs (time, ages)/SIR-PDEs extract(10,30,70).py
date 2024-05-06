# -*- coding: utf-8 -*-
"""
Created on Mon May  5 08:53:05 2024

@author: joonc
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
beta = 0.3  # Infection rate
gamma = 0.1  # Recovery rate

#Be aware of the Stability
h = 1  # Time step size
k = 1  # Age step size

# Time and age limits
T = 300  # Total time
A = 100  # Maximum age

# Discretization
num_t = int(T / h) + 1  # Number of time steps
num_a = int(A / k) + 1  # Number of age steps

# Initialize populations
S = np.zeros((num_a, num_t))
I = np.zeros((num_a, num_t))
R = np.zeros((num_a, num_t))

#Initial conditions
# Sigmoid function for initial susceptible population
def sigmoid(x):
    return 1 / (1 + np.exp(0.1 * (x - 80)))

# Apply sigmoid function to set initial susceptible population
for i in range(num_a):
    S[i, 0] = sigmoid(i * k)

# S[:, 0] = 1  # Initial susceptible population

# S[:20, 0] = 0.2  # Initial susceptible population
# S[20:50, 0] = 0.3  # Initial susceptible population
# S[50:, 0] = 0.3  # Initial susceptible population

I[:, 0] = 0.0001 #Initial infected population (initiated all age groups)


#I[20, 0] = 0.0001    # Initial infected population
R[:, 0] = 0     # Initial recovered population

# Finite difference method to update the populations
for j in range(num_t - 1):  # Time loop
    for i in range(1, num_a):  # Start from 1, Age loop
        S[i, j+1] = S[i-1, j] - beta*S[i, j]*I[i, j]*h
        I[i, j+1] = I[i-1, j] + beta*S[i, j]*I[i, j]*h - 2*gamma*I[i, j]*h 
        R[i, j+1] = R[i-1, j] + gamma * I[i, j] * h 
    
    # Update the 0th row to be equal to the 1st row for each variable
    S[0, j+1] = S[1, j+1]
    I[0, j+1] = I[1, j+1]
    R[0, j+1] = R[1, j+1]
    
    
# # Finite difference method to update the populations
# for j in range(num_t - 1):  # Time loop
#     for i in range(1, num_a):  # Start from 1, Age loop
#         S[i, j+1] = S[i-1, j] - 2*beta*S[i, j]*I[i, j]*h
#         I[i, j+1] = I[i-1, j] + 2*beta*S[i, j]*I[i, j]*h - 2*gamma*I[i, j]*h 
#         R[i, j+1] = R[i-1, j] + 2*gamma * I[i, j] * h 
    
#     # Update the 0th row to be equal to the 1st row for each variable
#     S[0, j+1] = S[1, j+1]
#     I[0, j+1] = I[1, j+1]
#     R[0, j+1] = R[1, j+1]
        
# # Finite difference method to update the populations
# for j in range(num_t-1):  # Time loop
#     for i in range(0, num_a-1):  # Age loop
#         S[i, j+1] = S[i+1, j] - beta*S[i, j]*I[i, j]*h
#         I[i, j+1] = I[i+1, j] + beta*S[i, j]*I[i, j]*h - gamma*I[i, j]*h 
#         R[i, j+1] = R[i+1, j] + gamma * I[i, j] * h

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
# Assume S, I, R matrices are already populated by your SIR model

# Define the ages you are interested in
age_indices = [10, 30, 70]  # Corresponding to ages 10, 30, and 70

# Time array
time_array = np.arange(num_t)

# Create a figure for 3D plots
fig, axs = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(18, 6))

# Titles for each subplot
titles = ['Susceptible Populations', 'Infected Populations', 'Recovered Populations']
data_arrays = [S, I, R]
colors = ['blue', 'red', 'green']  # Colors for different age lines

# Plotting
for ax, data, title in zip(axs, data_arrays, titles):
    for age_index, color in zip(age_indices, colors):
        ax.plot(time_array, data[age_index, :], zs=age_index, zdir='y', label=f'Age {age_index}', color=color)

    ax.set_title(title)
    ax.set_xlabel('Days')
    ax.set_ylabel('Age')
    ax.set_zlabel('Population')
    ax.legend()

plt.tight_layout()
plt.show()
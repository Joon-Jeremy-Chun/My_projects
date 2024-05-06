# -*- coding: utf-8 -*-
"""
Created on Mon May  4 07:39:24 2024

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
#%%

# Time and age for plotting
T_grid, A_grid = np.meshgrid(np.arange(num_t), np.arange(num_a))

# Plotting
fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(A_grid, T_grid, S, cmap='viridis')
ax1.set_title('Susceptible Populations')
ax1.set_xlabel('Age')
ax1.set_ylabel('Days')
ax1.set_zlabel('Population')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(A_grid, T_grid, I, cmap='viridis')
ax2.set_title('Infected Populations')
ax2.set_xlabel('Age')
ax2.set_ylabel('Days')
ax2.set_zlabel('Population')
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(A_grid, T_grid, R, cmap='viridis')
ax3.set_title('Recovered Populations')
ax3.set_xlabel('Age')
ax3.set_ylabel('Days')
ax3.set_zlabel('Population')
fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# #%%
# # Time and age for plotting
# T_grid, A_grid = np.meshgrid(np.arange(num_t), np.arange(num_a))

# # Plotting
# fig = plt.figure(figsize=(18, 6))

# ax1 = fig.add_subplot(131, projection='3d')
# surf1 = ax1.plot_surface(A_grid, T_grid, S, cmap='viridis')
# ax1.view_init(elev= -150, azim = 30)  # Adjusted for 90-degree clockwise rotation
# ax1.set_title('Susceptible Populations')
# ax1.set_xlabel('Age')
# ax1.set_ylabel('Days')
# ax1.set_zlabel('Population')
# fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

# ax2 = fig.add_subplot(132, projection='3d')
# surf2 = ax2.plot_surface(A_grid, T_grid, I, cmap='viridis')
# ax2.view_init(elev= -150, azim = 30)  # Adjusted for 90-degree clockwise rotation
# ax2.set_title('Infected Populations')
# ax2.set_xlabel('Age')
# ax2.set_ylabel('Days')
# ax2.set_zlabel('Population')
# fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

# ax3 = fig.add_subplot(133, projection='3d')
# surf3 = ax3.plot_surface(A_grid, T_grid, R, cmap='viridis')
# ax3.view_init(elev= -150, azim = 30)  # Adjusted for 90-degree clockwise rotation
# ax3.set_title('Recovered Populations')
# ax3.set_xlabel('Age')
# ax3.set_ylabel('Days')
# ax3.set_zlabel('Population')
# fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)

# plt.tight_layout()
# plt.show()
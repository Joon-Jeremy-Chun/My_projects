# -*- coding: utf-8 -*-
"""
Created on Sun May  5 16:47:39 2024

@author: joonc
"""

import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

# Parameters
beta = 0.3  # infection rate
gamma = 0.1  # recovery rate
h = 1  # age step size
k = 0.01  # time step size
total_time = 200  # total time to simulate
total_age = 100  # total age range to simulate
M = int(total_age / h)  # number of age steps
N = int(total_time / k)  # number of time steps

# Initial conditions
S = np.zeros((M, N))
I = np.zeros((M, N))
R = np.zeros((M, N))
#S[:, 0] = 1  # Assuming all ages start susceptible except the youngest
#I[0, 0] = 1    # Initial infected population at youngest age
I[:, 0] = 0.001    # Initial infected population all age
R[:, 0] = 0    # Initial recovered population
#%%
#Initial conditions
# Sigmoid function for initial susceptible population
def sigmoid(x):
    return 1 / (1 + np.exp(0.1 * (x - 80)))

# Apply sigmoid function to set initial susceptible population
for i in range(M):
    S[i, 0] = sigmoid(i * h)
#%%
# Simulation with explicit finite differences
for j in range(0, N-1):
    for i in range(0, M):
        S[i, j+1] = - k*beta*S[i,j]*I[i,j] + ((h-k)/h)*S[i,j] + (k/h)*S[i-1,j]
        I[i, j+1] = + k*beta*S[i,j]*I[i,j] - k*gamma*I[i,j] + ((h-k)/h)* I[i,j] + (k/h)* I[i-1,j]
        R[i, j+1] = + k*gamma*I[i,j] + ((h-k)/h)*R[i,j] + (k/h)*R[i-1,j]

        # Boundary condition: Update the 0th row to be equal to the 1st row for each variable
        S[0, j+1] = S[1, j+1]
        I[0, j+1] = I[1, j+1]
        R[0, j+1] = R[1, j+1]


# for j in range(0, N-1):
#     for i in range(0, M):
#         S[i, j+1] = - k*beta*S[i,j]*I[i,j] + ((h+k)/h)*S[i,j] + (k/h)*S[i-1,j]
#         I[i, j+1] = + k*beta*S[i,j]*I[i,j] - k*gamma*I[i,j] + ((h+k)/h)* I[i,j] + (k/h)* I[i-1,j]
#         R[i, j+1] = + k*gamma*I[i,j] + ((h+k)/h)*R[i,j] + (k/h)*R[i-1,j]
#%%
z = -k*beta*S[i,j]
S[i,j]
I[i,j]
#%%
# Creating 3D plots
fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

time = np.linspace(0, total_time, N)
age = np.linspace(0, total_age, M)
Time, Age = np.meshgrid(time, age)

# Plotting
ax1.plot_surface(Time, Age, S, cmap='viridis')
ax1.set_title('Susceptible')
ax1.set_xlabel('Time')
ax1.set_ylabel('Age')
ax1.set_zlabel('Population')

ax2.plot_surface(Time, Age, I, cmap='inferno')
ax2.set_title('Infected')
ax2.set_xlabel('Time')
ax2.set_ylabel('Age')
ax2.set_zlabel('Population')

ax3.plot_surface(Time, Age, R, cmap='plasma')
ax3.set_title('Recovered')
ax3.set_xlabel('Time')
ax3.set_ylabel('Age')
ax3.set_zlabel('Population')

plt.tight_layout()
plt.show()

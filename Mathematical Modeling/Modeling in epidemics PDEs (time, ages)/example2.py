# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 20:54:30 2024

@author: joonc
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
beta = np.array([[0.2, 0.1],  # Transmission rates (row: from, col: to)
                 [0.1, 0.15]])
gamma = np.array([0.05, 0.05])  # Recovery rates for each age group
N = np.array([1000, 1000])  # Total population in each age group
I0 = np.array([1, 1])  # Initial number of infected individuals in each age group
S0 = N - I0  # Initial number of susceptible individuals
R0 = np.array([0, 0])  # Initial number of recovered individuals
dt = 0.1  # Time step
T = 100  # Total time
steps = int(T / dt)  # Number of time steps

# Initialization
S = np.zeros((2, steps))
I = np.zeros((2, steps))
R = np.zeros((2, steps))
S[:, 0] = S0
I[:, 0] = I0
R[:, 0] = R0

# Time integration using Euler's method
for t in range(1, steps):
    for a in range(2):  # Loop over age groups
        # Calculate new infections
        new_infections = 0
        for b in range(2):  # Loop over all age groups for interactions
            new_infections += beta[a, b] * S[a, t-1] * I[b, t-1] / N[b]

        # Update compartments
        S[a, t] = S[a, t-1] - new_infections * dt
        I[a, t] = I[a, t-1] + (new_infections - gamma[a] * I[a, t-1]) * dt
        R[a, t] = R[a, t-1] + gamma[a] * I[a, t-1] * dt

# Plotting
t = np.linspace(0, T, steps)
plt.figure(figsize=(12, 6))
for a in range(2):
    plt.plot(t, S[a, :], label=f'Susceptible Age Group {a+1}')
    plt.plot(t, I[a, :], label=f'Infected Age Group {a+1}')
    plt.plot(t, R[a, :], label=f'Recovered Age Group {a+1}')

plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Age-Structured SIR Model Simulation')
plt.legend()
plt.show()
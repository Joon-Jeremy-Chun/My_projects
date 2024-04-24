# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 22:02:25 2024

@author: joonc
"""
import numpy as np
import matplotlib.pyplot as plt

# Parameters
beta = 0.3   # Infection rate
gamma = 0.1     # Recovery rate
time_steps = 1000  # Total number of time steps
age_groups = 100  # Number of age groups
dt = 0.1         # Time step size
da = 1          # Age group width

# Initial population state
S = np.zeros((age_groups, time_steps))
I = np.zeros((age_groups, time_steps))
R = np.zeros((age_groups, time_steps))

# Initial conditions
S[:, 0] = 1000   # Initial susceptible population in each age group
I[0, 0] = 10     # Initial infected population in the youngest age group
R[:, 0] = 0      # Initial recovered population is zero

def update_population(S, I, R, beta, gamma, da, dt, age_groups, time_steps):
    # Update the population over the time steps
    for t in range(time_steps - 1):  # Ensure we don't go out of bounds
        for a in range(age_groups):
            if a == 0:
                # Apply boundary condition for the youngest age group
                # Here you might define special handling or use similar equations without looking back to non-existent younger age groups
                S[a, t + 1] = S[a, t -1] - beta * S[a, t] * I[a, t] * dt
                I[a, t + 1] = I[a, t -1] + (beta * S[a, t] * I[a, t] - gamma * I[a, t]) * dt
                R[a, t + 1] = R[a, t -1] + gamma * I[a, t] * dt
            else:
                # Update S, I, R based on PDE discretization
                S[a, t + 1] = S[a, t -1] - beta * S[a, t] * I[a, t] * dt
                I[a, t + 1] = I[a, t -1] + (beta * S[a, t] * I[a, t] - gamma * I[a, t]) * dt
                R[a, t + 1] = R[a, t -1] + gamma * I[a, t] * dt

    return S, I, R

# Run the model
S, I, R = update_population(S, I, R, beta, gamma, da, dt, age_groups, time_steps)

# Plot the results
def plot_results(S, I, R, age_groups, time_steps, dt):
    plt.figure(figsize=(14, 5))
    plt.subplot(131)
    plt.imshow(S, aspect='auto', origin='lower')
    plt.colorbar(label='Susceptible')
    plt.subplot(132)
    plt.imshow(I, aspect='auto', origin='lower')
    plt.colorbar(label='Infected')
    plt.subplot(133)
    plt.imshow(R, aspect='auto', origin='lower')
    plt.colorbar(label='Recovered')
    plt.tight_layout()
    plt.show()

plot_results(S, I, R, age_groups, time_steps, dt)

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 08:16:02 2024

@author: joonc

Description:
Modeling from 'A modified age-structured SIR model for COVID-19 type viruses' by Vishaal Ram and Laura P. Schaposnik.
This script implements an age-structured SIR model with three groups: children, adults, and seniors.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Group identifiers for zero-based indexing
CHILDREN, ADULTS, SENIORS = 0, 1, 2

def deriv(y, t, N, beta, gamma, M):
    """
    Compute the derivatives for the SIR model with age structure.

    Parameters:
    y (list): Flattened list of S, I, R values for each group.
    t (float): Time.
    N (dict): Total population for each group.
    beta (float): Transmission rate.
    gamma (dict): Recovery rates for each group.
    M (numpy array): Contact matrix between groups.

    Returns:
    list: Flattened list of derivatives for S, I, R values.
    """
    S = y[::3]  # Extract all Susceptible compartments
    I = y[1::3] # Extract all Infected compartments
    R = y[2::3] # Extract all Recovered compartments

    dSdt = [-beta * S[i] / N[i] * sum(M[i, j] * I[j] for j in range(len(N))) for i in range(len(N))]
    dIdt = [beta * S[i] / N[i] * sum(M[i, j] * I[j] for j in range(len(N))) - gamma[i] * I[i] for i in range(len(N))]
    dRdt = [gamma[i] * I[i] for i in range(len(N))]

    return sum([list(tup) for tup in zip(dSdt, dIdt, dRdt)], [])

# Initial conditions and parameters
group_initial_conditions = {
    CHILDREN: {'S0': 3000, 'I0': 1, 'R0': 0},
    ADULTS: {'S0': 3000, 'I0': 1, 'R0': 0},
    SENIORS: {'S0': 2000, 'I0': 1, 'R0': 0}
}

# Contact matrix (3x3 example)
M = np.array([
    [0.15, 0.3, 0.2],  # CHILDREN to CHILDREN, ADULTS, SENIORS
    [0.3, 0.35, 0.15], # ADULTS to CHILDREN, ADULTS, SENIORS
    [0.2, 0.15, 0.25]  # SENIORS to CHILDREN, ADULTS, SENIORS
])

# Transmission rate
beta_constant = 0.5  # Example value, set this to your chosen average contacts rate

# Recovery rates
gamma = {CHILDREN: 0.12, ADULTS: 0.1, SENIORS: 0.08}

# Total populations
N = {group: conditions['S0'] + conditions['I0'] + conditions['R0'] for group, conditions in group_initial_conditions.items()}

# Time grid (in days) and initial conditions vector
t = np.linspace(0, 400, 4000)
y0 = sum(([conditions[key] for key in ['S0', 'I0', 'R0']] for conditions in group_initial_conditions.values()), [])

# Integrate the SIR equations over the time grid
ret = odeint(deriv, y0, t, args=(N, beta_constant, gamma, M))

# Extract results
Sc, Ic, Rc, Sa, Ia, Ra, Ss, Is, Rs = ret.T

# Plotting
plt.figure(figsize=(14, 10))
for idx, group in enumerate([CHILDREN, ADULTS, SENIORS], start=1):
    plt.subplot(3, 1, idx)
    plt.plot(t, ret.T[3*(idx-1)], 'b', label=f'Susceptible {group}')
    plt.plot(t, ret.T[3*(idx-1)+1], 'r', label=f'Infected {group}')
    plt.plot(t, ret.T[3*(idx-1)+2], 'g', label=f'Recovered {group}')
    plt.title(f'SIR Model Dynamics for Group {group}')
    plt.xlabel('Days')
    plt.ylabel('Population')
    plt.legend()
plt.tight_layout()
plt.show()

# Create a new figure for the 3D plot
fig = plt.figure(figsize=(16, 12))

# 3D plot for Susceptible population
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(t, np.full_like(t, CHILDREN), Sc, label='Children', color='blue')
ax1.plot(t, np.full_like(t, ADULTS), Sa, label='Adults', color='red')
ax1.plot(t, np.full_like(t, SENIORS), Ss, label='Seniors', color='green')
ax1.set_title('Susceptible Populations')
ax1.set_xlabel('Days')
ax1.set_ylabel('Group')
ax1.set_zlabel('Population')
ax1.set_yticks([CHILDREN, ADULTS, SENIORS])
ax1.set_yticklabels(['Children', 'Adults', 'Seniors'])
ax1.legend()

# 3D plot for Infected population
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot(t, np.full_like(t, CHILDREN), Ic, label='Children', color='blue')
ax2.plot(t, np.full_like(t, ADULTS), Ia, label='Adults', color='red')
ax2.plot(t, np.full_like(t, SENIORS), Is, label='Seniors', color='green')
ax2.set_title('Infected Populations')
ax2.set_xlabel('Days')
ax2.set_ylabel('Group')
ax2.set_zlabel('Population')
ax2.set_yticks([CHILDREN, ADULTS, SENIORS])
ax2.set_yticklabels(['Children', 'Adults', 'Seniors'])
ax2.legend()

# 3D plot for Recovered population
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot(t, np.full_like(t, CHILDREN), Rc, label='Children', color='blue')
ax3.plot(t, np.full_like(t, ADULTS), Ra, label='Adults', color='red')
ax3.plot(t, np.full_like(t, SENIORS), Rs, label='Seniors', color='green')
ax3.set_title('Recovered Populations')
ax3.set_xlabel('Days')
ax3.set_ylabel('Group')
ax3.set_zlabel('Population')
ax3.set_yticks([CHILDREN, ADULTS, SENIORS])
ax3.set_yticklabels(['Children', 'Adults', 'Seniors'])
ax3.legend()

plt.tight_layout()
plt.show()

# Save the figure
fig.savefig("SIR_i_groups_model_3D_plots.png", dpi=600)

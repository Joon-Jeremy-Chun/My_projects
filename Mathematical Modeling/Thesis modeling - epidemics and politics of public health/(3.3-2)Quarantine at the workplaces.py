# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 22:22:12 2024

@author: joonc
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 21:58:42 2024

@author: joonc
"""

import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.signal import argrelextrema

# Function to load model parameters from Excel
def load_parameters(file_path):
    df = pd.read_excel(file_path, header=None)
    
    # Transmission rates matrix (3x3)
    transmission_rates = df.iloc[0:3, 0:3].values
    
    # Recovery rates (3 values)
    recovery_rates = df.iloc[4, 0:3].values
    
    # Maturity rates (2 values)
    maturity_rates = df.iloc[6, 0:2].values
    
    # Waning immunity rate (single value)
    waning_immunity_rate = df.iloc[8, 0]
    
    # Vaccination rates (3 values)
    vaccination_rates = df.iloc[10, 0:3].values
    
    # Time span (in days)
    time_span = df.iloc[12, 0]
    
    # Population sizes (3 values)
    population_size = df.iloc[14, 0:3].values
    
    return (transmission_rates, recovery_rates, maturity_rates, waning_immunity_rate,
            vaccination_rates, time_span, population_size)

# Load the parameters from the Excel file
file_path = 'Inputs.xlsx'
(beta, gamma, mu, W, a, time_span, N) = load_parameters(file_path)

# Group indices for readability
S1, I1, R1, V1 = 0, 1, 2, 3  # First group (e.g., children)

# SIRV model differential equations including vaccinated compartments
def deriv(y, t, N, beta, gamma, mu, W, a):
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y
    
    # Force of infection (λ_i) for each group
    lambda1 = beta[0, 0] * I1/N[0] + beta[0, 1] * I2/N[1] + beta[0, 2] * I3/N[2]
    lambda2 = beta[1, 0] * I1/N[0] + beta[1, 1] * I2/N[1] + beta[1, 2] * I3/N[2]
    lambda3 = beta[2, 0] * I1/N[0] + beta[2, 1] * I2/N[1] + beta[2, 2] * I3/N[2]
    
    # Differential equations for each compartment
    dS1dt = -lambda1 * S1 - a[0] * S1 + W * R1 + W * V1
    dI1dt = lambda1 * S1 - gamma[0] * I1 - mu[0] * I1
    dR1dt = gamma[0] * I1 - W * R1 - mu[0] * R1
    dV1dt = a[0] * S1 - W * V1
    
    dS2dt = -lambda2 * S2 + mu[0] * S1 - a[1] * S2 + W * R2 + W * V2
    dI2dt = lambda2 * S2 + mu[0] * I1 - gamma[1] * I2 - mu[1] * I2
    dR2dt = gamma[1] * I2 + mu[0] * R1 - W * R2 - mu[1] * R2
    dV2dt = a[1] * S2 - W * V2
    
    dS3dt = -lambda3 * S3 + mu[1] * S2 - a[2] * S3 + W * R3 + W * V3
    dI3dt = lambda3 * S3 + mu[1] * I2 - gamma[2] * I3
    dR3dt = gamma[2] * I3 + mu[1] * R2 - W * R3
    dV3dt = a[2] * S3 - W * V3
    
    return dS1dt, dI1dt, dR1dt, dV1dt, dS2dt, dI2dt, dR2dt, dV2dt, dS3dt, dI3dt, dR3dt, dV3dt

# Function to find the local maxima
def find_local_maxima(I):
    maxima_indices = argrelextrema(I, np.greater)[0]
    maxima_values = I[maxima_indices]
    return maxima_indices, maxima_values

# Initial conditions for the three groups (including vaccinated compartments)
initial_conditions = [
    N[0] - 0.0001 * N[0], 0.0001 * N[0], 0, 0,  # S1, I1, R1, V1 (Group 1)
    N[1] - 0.0001 * N[1], 0.0001 * N[1], 0, 0,  # S2, I2, R2, V2 (Group 2)
    N[2] - 0.0001 * N[2], 0.0001 * N[2], 0, 0   # S3, I3, R3, V3 (Group 3)
]

# Time grid (in days)
t = np.linspace(0, time_span, int(time_span))

# Set up color maps for each group
colors_group1 = cm.get_cmap('Reds', 10)    # Red for Group 1 (Children)
colors_group2 = cm.get_cmap('Greens', 10)  # Green for Group 2 (Adults)
colors_group3 = cm.get_cmap('Blues', 10)   # Blue for Group 3 (Seniors)

# Create a figure for each group
fig, axes = plt.subplots(3, 1, figsize=(10, 18))

# Group 1 (Children)
for i in range(10):
    beta[1, 1] *= 0.9  # Now reducing β22 instead of β11 for quarantine levels at workplace
    results = odeint(deriv, initial_conditions, t, args=(N, beta, gamma, mu, W, a))
    _, I1, _, _, _, _, _, _, _, _, _, _ = results.T
    axes[0].plot(t, I1, color=colors_group1(i), label=f'Quarantine Level {i+1}: β22={beta[1,1]:.4f}')
    maxima_indices, maxima_values = find_local_maxima(I1)
    axes[0].scatter(t[maxima_indices], maxima_values, color='black', zorder=5)
    for idx, val in zip(maxima_indices, maxima_values):
        axes[0].annotate(f'({t[idx]:.1f}, {val:.1f})', (t[idx], val), textcoords="offset points", xytext=(0, 10), ha='center')
axes[0].set_title('Infectious Population (Group 1: Children)')
axes[0].set_xlabel('Days')
axes[0].set_ylabel('Infectious Population')
axes[0].legend(loc='upper right')

# Reset beta for next group
(beta, gamma, mu, W, a, time_span, N) = load_parameters(file_path)

# Group 2 (Adults)
for i in range(10):
    beta[1, 1] *= 0.9
    results = odeint(deriv, initial_conditions, t, args=(N, beta, gamma, mu, W, a))
    _, _, _, _, _, I2, _, _, _, _, _, _ = results.T
    axes[1].plot(t, I2, color=colors_group2(i), label=f'Quarantine Level {i+1}: β22={beta[1,1]:.4f}')
    maxima_indices, maxima_values = find_local_maxima(I2)
    axes[1].scatter(t[maxima_indices], maxima_values, color='black', zorder=5)
    for idx, val in zip(maxima_indices, maxima_values):
        axes[1].annotate(f'({t[idx]:.1f}, {val:.1f})', (t[idx], val), textcoords="offset points", xytext=(0, 10), ha='center')
axes[1].set_title('Infectious Population (Group 2: Adults)')
axes[1].set_xlabel('Days')
axes[1].set_ylabel('Infectious Population')
axes[1].legend(loc='upper right')

# Reset beta for next group
(beta, gamma, mu, W, a, time_span, N) = load_parameters(file_path)

# Group 3 (Seniors)
for i in range(10):
    beta[1, 1] *= 0.9
    results = odeint(deriv, initial_conditions, t, args=(N, beta, gamma, mu, W, a))
    _, _, _, _, _, _, _, _, _, I3, _, _ = results.T
    axes[2].plot(t, I3, color=colors_group3(i), label=f'Quarantine Level {i+1}: β22={beta[1,1]:.4f}')
    maxima_indices, maxima_values = find_local_maxima(I3)
    axes[2].scatter(t[maxima_indices], maxima_values, color='black', zorder=5)
    for idx, val in zip(maxima_indices, maxima_values):
        axes[2].annotate(f'({t[idx]:.1f}, {val:.1f})', (t[idx], val), textcoords="offset points", xytext=(0, 10), ha='center')
axes[2].set_title('Infectious Population (Group 3: Seniors)')
axes[2].set_xlabel('Days')
axes[2].set_ylabel('Infectious Population')
axes[2].legend(loc='upper right')

# Show the plot for all three groups
plt.tight_layout()
plt.show()
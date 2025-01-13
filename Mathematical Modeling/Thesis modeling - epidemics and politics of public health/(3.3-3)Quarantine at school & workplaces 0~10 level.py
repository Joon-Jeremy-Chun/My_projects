# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 23:50:12 2024

@author: joonc
"""
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
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

# Function to create independent plots for each group
def plot_group_infected(group_idx, group_label, beta, N, gamma, mu, W, a, initial_conditions):
    # Create a figure for the plot
    plt.figure(figsize=(12, 8))
    colors = plt.get_cmap('coolwarm', 10)  # Color map for mixed policies

    # Loop to reduce β11 (school quarantine) and β22 (workplace quarantine) and plot the Infected curves for the selected group
    for i in range(10):
        # Reduce β11 (school) and β22 (workplace) by 10% each time
        beta[0, 0] *= 0.9
        beta[1, 1] *= 0.9

        # Integrate the SIRV equations over time
        results = odeint(deriv, initial_conditions, t, args=(N, beta, gamma, mu, W, a))

        # Extract results for the Infected group
        I = results[:, group_idx + 1]  # Index for the infected group

        # Plot the Infected population for the group with different colors for each mixed policy level
        plt.plot(t, I, color=colors(i), label=f'Level {i+1}: β11={beta[0,0]:.4f}, β22={beta[1,1]:.4f}')

        # Find and plot local maxima
        maxima_indices, maxima_values = find_local_maxima(I)
        plt.scatter(t[maxima_indices], maxima_values, color='black', zorder=5)  # Black dots for maxima

        # Annotate maxima with coordinates
        for idx, val in zip(maxima_indices, maxima_values):
            plt.annotate(f'({t[idx]:.1f}, {val:.1f})', (t[idx], val), textcoords="offset points", xytext=(0, 10), ha='center')

    # Add labels, title, and legend
    plt.title(f'Infectious Population for {group_label} under Different Quarantine Levels with Maxima')
    plt.xlabel('Days')
    plt.ylabel(f'Infectious Population ({group_label})')
    plt.legend(loc='upper right')

    # Show the plot
    plt.tight_layout()
    plt.show()

# Plot for Group 1 (Children)
plot_group_infected(0, "Group 1 (Children)", beta.copy(), N, gamma, mu, W, a, initial_conditions)

# Plot for Group 2 (Adults)
plot_group_infected(4, "Group 2 (Adults)", beta.copy(), N, gamma, mu, W, a, initial_conditions)

# Plot for Group 3 (Seniors)
plot_group_infected(8, "Group 3 (Seniors)", beta.copy(), N, gamma, mu, W, a, initial_conditions)

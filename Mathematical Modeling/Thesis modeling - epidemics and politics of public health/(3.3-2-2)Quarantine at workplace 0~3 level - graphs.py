# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 08:50:23 2024

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
    
    # Initial conditions: Susceptible, Infectious, Recovered, Vaccinated (each for 3 groups)
    susceptible_init = df.iloc[14, 0:3].values  # Define population size
    infectious_init = df.iloc[16, 0:3].values  
    recovered_init = df.iloc[18, 0:3].values   
    vaccinated_init = df.iloc[20, 0:3].values  
    
    return (recovery_rates, maturity_rates, waning_immunity_rate,
            vaccination_rates, time_span, population_size, susceptible_init,
            infectious_init, recovered_init, vaccinated_init)

# Load the parameters from the Excel file
file_path = 'Inputs.xlsx'
(gamma, mu, W, a, time_span, N, S_init, I_init, R_init, V_init) = load_parameters(file_path)

# Load the transmission_rates matrix from the CSV file
transmission_rates_csv_path = 'DataSets/Fitted_Beta_Matrix.csv'
transmission_rates = pd.read_csv(transmission_rates_csv_path).iloc[0:3, 1:4].values

# Override the beta parameter with the loaded transmission_rates
beta = transmission_rates


# Define initial conditions for the three groups (including vaccinated compartments)
initial_conditions = [
    S_init[0], I_init[0], R_init[0], V_init[0],  # S1, I1, R1, V1 (Group 1)
    S_init[1], I_init[1], R_init[1], V_init[1],  # S2, I2, R2, V2 (Group 2)
    S_init[2], I_init[2], R_init[2], V_init[2]   # S3, I3, R3, V3 (Group 3)
]

# SIRV model differential equations including vaccinated compartments
def deriv(y, t, N, beta, gamma, mu, W, a):
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y
    
    # Force of infection (\u03bb_i) for each group
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

# Function to compute reduction in peak and delay in peak for all groups and total population
def compute_peak_reduction_and_delay_all(quarantine_levels, beta, N, gamma, mu, W, a, initial_conditions, time_span):
    results = {'Group 1': [], 'Group 2': [], 'Group 3': [], 'Total': []}
    t = np.linspace(0, time_span, int(time_span))

    # Simulate level 0 (no quarantine)
    adjusted_beta = beta.copy()
    results_level0 = odeint(deriv, initial_conditions, t, args=(N, adjusted_beta, gamma, mu, W, a))
    I1_level0, I2_level0, I3_level0 = results_level0[:, 1], results_level0[:, 5], results_level0[:, 9]
    total_infected_level0 = I1_level0 + I2_level0 + I3_level0

    level0_peaks = {
        'Group 1': find_local_maxima(I1_level0),
        'Group 2': find_local_maxima(I2_level0),
        'Group 3': find_local_maxima(I3_level0),
        'Total': find_local_maxima(total_infected_level0)
    }

    # Iterate through each level
    for level in quarantine_levels[1:]:
        adjusted_beta = beta.copy()
        # adjusted_beta[0, 0] *= (1 - level)  
        # adjusted_beta[0, 1] *= (1 - level)
        # adjusted_beta[0, 2] *= (1 - level)
        # adjusted_beta[1, 0] *= (1 - level)
        adjusted_beta[1, 1] *= (1 - level)         #Beta [1,1]
        # adjusted_beta[1, 2] *= (1 - level)
        # adjusted_beta[2, 0] *= (1 - level)
        # adjusted_beta[2, 1] *= (1 - level)
        # adjusted_beta[2, 2] *= (1 - level)
        
        
        results_current = odeint(deriv, initial_conditions, t, args=(N, adjusted_beta, gamma, mu, W, a))

        # Extract results for the Infected groups
        I1, I2, I3 = results_current[:, 1], results_current[:, 5], results_current[:, 9]
        total_infected = I1 + I2 + I3  # Sum of infected from all groups

        current_peaks = {
            'Group 1': find_local_maxima(I1),
            'Group 2': find_local_maxima(I2),
            'Group 3': find_local_maxima(I3),
            'Total': find_local_maxima(total_infected)
        }

        # Calculate the percentage reduction in peak and delay in days for each group and total
        for key in level0_peaks:
            level0_max_idx, level0_max_val = level0_peaks[key]
            max_idx, max_val = current_peaks[key]

            peak_level0 = level0_max_val[0] if len(level0_max_val) > 0 else 0
            peak_day_level0 = t[level0_max_idx[0]] if len(level0_max_idx) > 0 else 0

            peak_current = max_val[0] if len(max_val) > 0 else 0
            peak_day_current = t[max_idx[0]] if len(max_idx) > 0 else 0

            peak_reduction = ((peak_level0 - peak_current) / peak_level0) * 100 if peak_level0 > 0 else 0
            peak_delay = peak_day_current - peak_day_level0

            results[key].append({
                'quarantine_level': level,
                'peak_reduction_percent': peak_reduction,
                'peak_delay_days': peak_delay
            })

    return results

# Define quarantine levels
quarantine_levels = [0, 0.1, 0.35, 0.7]

# Compute peak reduction and delay for all groups and total population
peak_results_all = compute_peak_reduction_and_delay_all(quarantine_levels, beta, N, gamma, mu, W, a, initial_conditions, time_span)

# Print results grouped by group
for group, results in peak_results_all.items():
    print(f"Results for {group}:")
    for result in results:
        print(f"  Quarantine Level: {int(result['quarantine_level'] * 100)}% Reduction")
        print(f"  Peak Reduction: {result['peak_reduction_percent']:.2f}%")
        print(f"  Peak Delay: {result['peak_delay_days']:.2f} days")
    print()

# Function to create plots for each group with 4 quarantine levels
def plot_group_infected_4_levels_workplace(group_idx, group_label, beta, N, gamma, mu, W, a, initial_conditions, time_span):
    t = np.linspace(0, time_span, int(time_span))
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'orange', 'red']  # Colors for the 4 levels

    # Quarantine levels with respective reductions
    quarantine_levels = [0, 0.1, 0.35, 0.7]

    # Simulate Level 0 (no quarantine) to calculate baseline peak and timing
    results_base = odeint(deriv, initial_conditions, t, args=(N, beta, gamma, mu, W, a))
    I_base = results_base[:, group_idx + 1]
    peak_base = np.max(I_base)
    peak_time_base = t[np.argmax(I_base)]  # Time of peak for Level 0

    for i, level in enumerate(quarantine_levels):
        # Adjust beta[1,1] for the current quarantine level
        adjusted_beta = beta.copy()
        adjusted_beta[1, 1] *= (1 - level)  # Reduce beta for workplace transmission rate

        # Simulate the model
        results = odeint(deriv, initial_conditions, t, args=(N, adjusted_beta, gamma, mu, W, a))

        # Extract results for the Infected group
        I = results[:, group_idx + 1]
        maxima_indices, _ = find_local_maxima(I)

        # Plot the Infected population for the group
        plt.plot(t, I, color=colors[i], label=f'Level: {int(level * 100)}%, Peak Reduction: {((peak_base - np.max(I)) / peak_base) * 100:.2f}%, Delay: {t[np.argmax(I)] - peak_time_base:.2f} days')

        # Add black dots for relative maxima
        plt.scatter(t[maxima_indices], I[maxima_indices], color='black', zorder=5)

    plt.title(f'Infectious Population for {group_label} under Workplace Quarantine Levels')
    plt.xlabel('Days')
    plt.ylabel(f'Infectious Population ({group_label})')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


# Function to plot the total infected population over time for all age groups combined
def plot_total_infected_4_levels_workplace(beta, N, gamma, mu, W, a, initial_conditions, time_span):
    t = np.linspace(0, time_span, int(time_span))
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'orange', 'red']  # Colors for the 4 levels

    # Quarantine levels with respective reductions
    quarantine_levels = [0, 0.1, 0.35, 0.7]

    # Simulate Level 0 (no quarantine) to calculate baseline peak and timing
    results_base = odeint(deriv, initial_conditions, t, args=(N, beta, gamma, mu, W, a))
    total_base = results_base[:, 1] + results_base[:, 5] + results_base[:, 9]
    peak_base = np.max(total_base)
    peak_time_base = t[np.argmax(total_base)]  # Time of peak for Level 0

    for i, level in enumerate(quarantine_levels):
        # Adjust beta[1,1] for the current quarantine level
        adjusted_beta = beta.copy()
        adjusted_beta[1, 1] *= (1 - level)  # Reduce beta for workplace transmission rate

        # Simulate the model
        results = odeint(deriv, initial_conditions, t, args=(N, adjusted_beta, gamma, mu, W, a))

        # Extract results for the total Infected population
        total_infected = results[:, 1] + results[:, 5] + results[:, 9]
        maxima_indices, _ = find_local_maxima(total_infected)

        # Plot the total Infected population
        plt.plot(t, total_infected, color=colors[i], label=f'Level: {int(level * 100)}%, Peak Reduction: {((peak_base - np.max(total_infected)) / peak_base) * 100:.2f}%, Delay: {t[np.argmax(total_infected)] - peak_time_base:.2f} days')

        # Add black dots for relative maxima
        plt.scatter(t[maxima_indices], total_infected[maxima_indices], color='black', zorder=5)

    plt.title('Total Infectious Population under Workplace Quarantine Levels')
    plt.xlabel('Days')
    plt.ylabel('Total Infectious Population (All Groups)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# Plot for Group 1 (Children)
plot_group_infected_4_levels_workplace(0, "Group 1 (Children)", beta.copy(), N, gamma, mu, W, a, initial_conditions, time_span)

# Plot for Group 2 (Adults)
plot_group_infected_4_levels_workplace(4, "Group 2 (Adults)", beta.copy(), N, gamma, mu, W, a, initial_conditions, time_span)

# Plot for Group 3 (Seniors)
plot_group_infected_4_levels_workplace(8, "Group 3 (Seniors)", beta.copy(), N, gamma, mu, W, a, initial_conditions, time_span)

# Plot the total infected population for workplace quarantine levels
plot_total_infected_4_levels_workplace(beta.copy(), N, gamma, mu, W, a, initial_conditions, time_span)

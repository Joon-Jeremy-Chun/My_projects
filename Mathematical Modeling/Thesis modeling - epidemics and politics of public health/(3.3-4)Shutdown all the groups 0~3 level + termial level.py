# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 20:25:15 2025

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
def deriv(y, t, N, beta, gamma, mu, W, vaccination_rates):
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y
    
    # Force of infection (λ_i) for each group
    lambda1 = beta[0, 0] * I1/N[0] + beta[0, 1] * I2/N[1] + beta[0, 2] * I3/N[2]
    lambda2 = beta[1, 0] * I1/N[0] + beta[1, 1] * I2/N[1] + beta[1, 2] * I3/N[2]
    lambda3 = beta[2, 0] * I1/N[0] + beta[2, 1] * I2/N[1] + beta[2, 2] * I3/N[2]
    
    # Differential equations for each compartment
    dS1dt = -lambda1 * S1 - vaccination_rates[0] / N[0] * S1 + W * R1 + W * V1
    dI1dt = lambda1 * S1 - gamma[0] * I1 - mu[0] * I1
    dR1dt = gamma[0] * I1 - W * R1 - mu[0] * R1
    dV1dt = vaccination_rates[0] / N[0] * S1 - W * V1
    
    dS2dt = -lambda2 * S2 + mu[0] * S1 - vaccination_rates[1] / N[1] * S2 + W * R2 + W * V2
    dI2dt = lambda2 * S2 + mu[0] * I1 - gamma[1] * I2 - mu[1] * I2
    dR2dt = gamma[1] * I2 + mu[0] * R1 - W * R2 - mu[1] * R2
    dV2dt = vaccination_rates[1] / N[1] * S2 - W * V2
    
    dS3dt = -lambda3 * S3 + mu[1] * S2 - vaccination_rates[2] / N[2] * S3 + W * R3 + W * V3
    dI3dt = lambda3 * S3 + mu[1] * I2 - gamma[2] * I3
    dR3dt = gamma[2] * I3 + mu[1] * R2 - W * R3
    dV3dt = vaccination_rates[2] / N[2] * S3 - W * V3
    
    return dS1dt, dI1dt, dR1dt, dV1dt, dS2dt, dI2dt, dR2dt, dV2dt, dS3dt, dI3dt, dR3dt, dV3dt

# Function to find the local maxima
def find_local_maxima(I):
    maxima_indices = argrelextrema(I, np.greater)[0]
    maxima_values = I[maxima_indices]
    return maxima_indices, maxima_values

# Function to compute reduction in peak and delay for all groups and total population
def compute_peak_reduction_and_delay_all(quarantine_levels, beta, N, gamma, mu, W, a, initial_conditions, time_span):
    results = {'Group 1': [], 'Group 2': [], 'Group 3': [], 'Total': []}
    t = np.linspace(0, time_span, int(time_span))
    
    # Simulate baseline (Level 0, no quarantine)
    results_base = odeint(deriv, initial_conditions, t, args=(N, beta, gamma, mu, W, a))
    I1_level0, I2_level0, I3_level0 = results_base[:, 1], results_base[:, 5], results_base[:, 9]
    total_base = I1_level0 + I2_level0 + I3_level0
    
    level0_peaks = {}
    for key, data in zip(['Group 1', 'Group 2', 'Group 3', 'Total'],
                         [I1_level0, I2_level0, I3_level0, total_base]):
        max_idx, max_val = find_local_maxima(data)
        if len(max_idx) > 0 and max_idx[0] > 0:
            level0_peaks[key] = (max_idx, max_val)
        else:
            level0_peaks[key] = ([0], [data[0]])
    
    # Iterate through each quarantine level (excluding baseline)
    for level in quarantine_levels[1:]:
        adjusted_beta = beta.copy() * (1 - level)
        results_current = odeint(deriv, initial_conditions, t, args=(N, adjusted_beta, gamma, mu, W, a))
        I1, I2, I3 = results_current[:, 1], results_current[:, 5], results_current[:, 9]
        total_infected = I1 + I2 + I3
        
        current_peaks = {}
        for key, data in zip(['Group 1', 'Group 2', 'Group 3', 'Total'],
                              [I1, I2, I3, total_infected]):
            max_idx, max_val = find_local_maxima(data)
            if len(max_idx) == 0 or max_idx[0] == 0:
                current_peaks[key] = ([0], [data[0]])
            else:
                current_peaks[key] = (max_idx, max_val)
        
        # Compute metrics for each key
        for key in level0_peaks:
            level0_max_idx, level0_max_val = level0_peaks[key]
            current_max_idx, current_max_val = current_peaks[key]
            
            baseline_peak = level0_max_val[0]
            baseline_peak_day = t[level0_max_idx[0]]
            
            if len(current_max_idx) == 0 or current_max_idx[0] == 0:
                peak_reduction = "None"
                peak_delay = "None"
            else:
                current_peak = current_max_val[0]
                current_peak_day = t[current_max_idx[0]]
                peak_reduction = ((baseline_peak - current_peak) / baseline_peak) * 100 if baseline_peak > 0 else 0
                peak_delay = current_peak_day - baseline_peak_day
                # If delay is negative, set to None
                if peak_delay < 0:
                    peak_delay = "None"
            
            results[key].append({
                'quarantine_level': level,
                'peak_reduction_percent': peak_reduction,
                'peak_delay_days': peak_delay
            })
    
    return results

# Define quarantine levels
quarantine_levels = [0, 0.1, 0.35, 0.7, 0.74]

# Rainbow color scheme
colors = plt.cm.rainbow(np.linspace(0, 1, len(quarantine_levels)))

# Function to plot the total infected population over time
def plot_total_infected_all_levels(beta, N, gamma, mu, W, a, initial_conditions, time_span):
    t = np.linspace(0, time_span, int(time_span))
    plt.figure(figsize=(10, 6))
    
    # Simulate baseline (Level 0, no quarantine)
    results_base = odeint(deriv, initial_conditions, t, args=(N, beta, gamma, mu, W, a))
    total_base = results_base[:, 1] + results_base[:, 5] + results_base[:, 9]
    peak_base = np.max(total_base)
    peak_time_base = t[np.argmax(total_base)]
    
    for i, level in enumerate(quarantine_levels):
        adjusted_beta = beta.copy() * (1 - level)
        results = odeint(deriv, initial_conditions, t, args=(N, adjusted_beta, gamma, mu, W, a))
        total_infected = results[:, 1] + results[:, 5] + results[:, 9]
        max_idx, _ = find_local_maxima(total_infected)
        
        if len(max_idx) == 0 or max_idx[0] == 0:
            delay = "None"
        else:
            delay_val = t[max_idx[0]] - peak_time_base
            delay = "None" if delay_val < 0 else f"{delay_val:.2f}"
        
        plt.plot(t, total_infected, color=colors[i],
                 label=f'Level: {int(level * 100)}%, Peak Reduction: {((peak_base - np.max(total_infected)) / peak_base) * 100:.2f}%, Delay: {delay} days')
        idx, _ = find_local_maxima(total_infected)
        plt.scatter(t[idx], total_infected[idx], color='black', zorder=5)
    
    plt.title('Total Infectious Population under Population-Wide Quarantine Levels')
    plt.xlabel('Days')
    plt.ylabel('Total Infectious Population (All Groups)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# Compute peak reduction and delay for all groups and total population
peak_results_all = compute_peak_reduction_and_delay_all(quarantine_levels, beta, N, gamma, mu, W, a, initial_conditions, time_span)

# Display results for Total
for result in peak_results_all['Total']:
    print(f"Quarantine Level: {int(result['quarantine_level'] * 100)}% Reduction")
    print(f"  Peak Reduction: {result['peak_reduction_percent']}%")
    print(f"  Peak Delay: {result['peak_delay_days']} days\n")

# Plot the total infected population
plot_total_infected_all_levels(beta.copy(), N, gamma, mu, W, a, initial_conditions, time_span)

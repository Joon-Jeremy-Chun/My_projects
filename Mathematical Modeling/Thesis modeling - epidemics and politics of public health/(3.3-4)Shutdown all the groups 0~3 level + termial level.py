# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 20:25:15 2025

@author: joonc
"""

import os
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# Create "Figures" directory if it doesn't exist
if not os.path.exists("Figures"):
    os.makedirs("Figures")

# Global quarantine levels (applied uniformly to the entire beta matrix)
# For example, a level of 0.3 means a 30% reduction in all interactions,
# yielding a multiplier of 0.7.
quarantine_levels = [0, 0.3, 0.65, 0.8]

def load_parameters(file_path):
    """
    Load model parameters from an Excel file.
    """
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
    
    # Initial conditions: S, I, R, V for each of 3 groups
    susceptible_init = df.iloc[14, 0:3].values  
    infectious_init = df.iloc[16, 0:3].values  
    recovered_init = df.iloc[18, 0:3].values   
    vaccinated_init = df.iloc[20, 0:3].values  
    
    return (recovery_rates, maturity_rates, waning_immunity_rate,
            vaccination_rates, time_span, population_size,
            susceptible_init, infectious_init, recovered_init, vaccinated_init)

# Load parameters from Excel
file_path = 'Inputs.xlsx'
(gamma, mu, W, a, time_span, N, S_init, I_init, R_init, V_init) = load_parameters(file_path)

# Load the fitted beta matrix from CSV
transmission_rates_csv_path = 'DataSets/Fitted_Beta_Matrix.csv'
transmission_rates = pd.read_csv(transmission_rates_csv_path).iloc[0:3, 1:4].values

# Use the loaded transmission rates as beta
beta = transmission_rates

# Define initial conditions for the three groups (S, I, R, V each)
initial_conditions = [
    S_init[0], I_init[0], R_init[0], V_init[0],  # Group 1 (Children)
    S_init[1], I_init[1], R_init[1], V_init[1],  # Group 2 (Adults)
    S_init[2], I_init[2], R_init[2], V_init[2]   # Group 3 (Seniors)
]

def deriv(y, t, N, beta, gamma, mu, W, vaccination_rates):
    """
    SIRV model differential equations for 3 age groups.
    """
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y
    
    # Force of infection for each group
    lambda1 = beta[0, 0] * I1 / N[0] + beta[0, 1] * I2 / N[1] + beta[0, 2] * I3 / N[2]
    lambda2 = beta[1, 0] * I1 / N[0] + beta[1, 1] * I2 / N[1] + beta[1, 2] * I3 / N[2]
    lambda3 = beta[2, 0] * I1 / N[0] + beta[2, 1] * I2 / N[1] + beta[2, 2] * I3 / N[2]
    
    # Children
    dS1dt = -lambda1 * S1 - vaccination_rates[0] / N[0] * S1 + W * R1 + W * V1
    dI1dt = lambda1 * S1 - gamma[0] * I1 - mu[0] * I1
    dR1dt = gamma[0] * I1 - W * R1 - mu[0] * R1
    dV1dt = vaccination_rates[0] / N[0] * S1 - W * V1
    
    # Adults
    dS2dt = -lambda2 * S2 + mu[0] * S1 - vaccination_rates[1] / N[1] * S2 + W * R2 + W * V2
    dI2dt = lambda2 * S2 + mu[0] * I1 - gamma[1] * I2 - mu[1] * I2
    dR2dt = gamma[1] * I2 + mu[0] * R1 - W * R2 - mu[1] * R2
    dV2dt = vaccination_rates[1] / N[1] * S2 - W * V2
    
    # Seniors
    dS3dt = -lambda3 * S3 + mu[1] * S2 - vaccination_rates[2] / N[2] * S3 + W * R3 + W * V3
    dI3dt = lambda3 * S3 + mu[1] * I2 - gamma[2] * I3
    dR3dt = gamma[2] * I3 + mu[1] * R2 - W * R3
    dV3dt = vaccination_rates[2] / N[2] * S3 - W * V3
    
    return (dS1dt, dI1dt, dR1dt, dV1dt,
            dS2dt, dI2dt, dR2dt, dV2dt,
            dS3dt, dI3dt, dR3dt, dV3dt)

def find_local_maxima(I):
    """
    Find local maxima in a time series array I(t).
    Returns (maxima_indices, maxima_values).
    """
    maxima_indices = argrelextrema(I, np.greater)[0]
    maxima_values = I[maxima_indices]
    return maxima_indices, maxima_values

def compute_peak_reduction_and_delay_all_population(
    quarantine_levels, beta, N, gamma, mu, W, a, initial_conditions, time_span
):
    """
    Computes peak reduction and delay for population-wide quarantine
    by uniformly scaling the entire beta matrix by (1 - level).
    """
    results = {'Group 1': [], 'Group 2': [], 'Group 3': [], 'Total': []}
    t = np.linspace(0, time_span, int(time_span))
    
    # Baseline simulation (no quarantine, level 0)
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
    
    for level in quarantine_levels[1:]:
        # Uniformly scale the entire beta matrix
        adjusted_beta = beta * (1 - level)
        
        results_current = odeint(deriv, initial_conditions, t, args=(N, adjusted_beta, gamma, mu, W, a))
        I1, I2, I3 = results_current[:, 1], results_current[:, 5], results_current[:, 9]
        total_infected = I1 + I2 + I3
        
        current_peaks = {
            'Group 1': find_local_maxima(I1),
            'Group 2': find_local_maxima(I2),
            'Group 3': find_local_maxima(I3),
            'Total': find_local_maxima(total_infected)
        }
        
        # Calculate peak reduction and delay for each group
        for key in level0_peaks:
            level0_max_idx, level0_max_val = level0_peaks[key]
            max_idx, max_val = current_peaks[key]
            
            peak_level0 = level0_max_val[0] if len(level0_max_val) > 0 else 0
            peak_day_level0 = t[level0_max_idx[0]] if len(level0_max_idx) > 0 else 0
            
            peak_current = max_val[0] if len(max_val) > 0 else 0
            if len(max_idx) > 0:
                peak_day_current = t[max_idx[0]]
            else:
                peak_day_current = -999  # sentinel value if no maxima
            peak_reduction = ((peak_level0 - peak_current) / peak_level0) * 100 if peak_level0 > 0 else 0
            peak_delay = peak_day_current - peak_day_level0
            
            results[key].append({
                'quarantine_level': level,
                'peak_reduction_percent': peak_reduction,
                'peak_delay_days': peak_delay
            })
            
    return results

# --- MAIN COMPUTATION FOR SCENARIO 4 ---

peak_results_population = compute_peak_reduction_and_delay_all_population(
    quarantine_levels, beta, N, gamma, mu, W, a, initial_conditions, time_span
)

# Print results for population-wide scenario
for group, results in peak_results_population.items():
    print(f"Population-Wide Results for {group}:")
    for result in results:
        lvl = int(result['quarantine_level'] * 100)
        print(f"  Quarantine Level: {lvl}% Reduction")
        print(f"  Peak Reduction: {result['peak_reduction_percent']:.2f}%")
        
        # If there's no local max, 'peak_delay_days' might be negative or nonsense
        # We'll check if it's < 0 or a sentinel:
        if result['peak_delay_days'] < 0:
            print(f"  Peak Delay: None")
        else:
            print(f"  Peak Delay: {result['peak_delay_days']:.2f} days")
    print()

# --- PLOTTING FUNCTIONS ---

def plot_group_infected_4_levels_population(group_idx, group_label,
                                            beta, N, gamma, mu, W, a,
                                            initial_conditions, time_span):
    """
    Plots the infected population for a single age group under
    population-wide quarantine levels. If no local max is found,
    display 'None' in the legend.
    """
    t = np.linspace(0, time_span, int(time_span))
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'orange', 'red']
    
    # Baseline simulation (no quarantine)
    results_base = odeint(deriv, initial_conditions, t, args=(N, beta, gamma, mu, W, a))
    I_base = results_base[:, group_idx + 1]
    peak_base = np.max(I_base)
    peak_time_base = float(t[np.argmax(I_base)])
    
    for i, level in enumerate(quarantine_levels):
        # Uniformly scale the entire beta matrix
        adjusted_beta = beta * (1 - level)
        
        results = odeint(deriv, initial_conditions, t, args=(N, adjusted_beta, gamma, mu, W, a))
        I = results[:, group_idx + 1]
        
        # Find local maxima
        maxima_indices, _ = find_local_maxima(I)
        
        # Default placeholders if no local maxima
        peak_reduction_str = "None"
        peak_delay_str = "None"
        
        if len(maxima_indices) > 0:
            idx = maxima_indices[0]  # or pick the largest local max if you want
            peak_current = I[idx]
            peak_day_current = t[idx]
            
            # Compute peak reduction
            if peak_base > 0:
                peak_reduction_val = ((peak_base - peak_current) / peak_base) * 100
                peak_reduction_str = f"{peak_reduction_val:.2f}%"
            else:
                peak_reduction_str = "None"
            
            # Compute delay
            peak_delay_val = peak_day_current - peak_time_base
            if peak_delay_val >= 0:
                peak_delay_str = f"{peak_delay_val:.2f} days"
            else:
                peak_delay_str = "None"
        
        label_str = (f"Level: {int(level * 100)}% reduction "
                     f"({1 - level:.2f} multiplier), "
                     f"Peak Red: {peak_reduction_str}, "
                     f"Delay: {peak_delay_str}")
        
        plt.plot(t, I, color=colors[i], label=label_str)
        # Mark local maxima with black dots
        plt.scatter(t[maxima_indices], I[maxima_indices], color='black', zorder=5)
    
    plt.title(f'Infectious Population for {group_label} under Population-Wide Quarantine')
    plt.xlabel('Days')
    plt.ylabel(f'Infectious Population ({group_label})')
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save the figure
    filename = f"Figures/population_quarantine_{group_label.replace(' ', '_')}_Infected.png"
    plt.savefig(filename)
    plt.show()

def plot_total_infected_4_levels_population(beta, N, gamma, mu, W, a,
                                            initial_conditions, time_span):
    """
    Plots the total infected population under population-wide quarantine levels.
    If no local maxima is found, display 'None' in the legend.
    """
    t = np.linspace(0, time_span, int(time_span))
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'orange', 'red']
    
    # Baseline simulation (no quarantine)
    results_base = odeint(deriv, initial_conditions, t, args=(N, beta, gamma, mu, W, a))
    total_base = results_base[:, 1] + results_base[:, 5] + results_base[:, 9]
    peak_base = np.max(total_base)
    peak_time_base = float(t[np.argmax(total_base)])
    
    for i, level in enumerate(quarantine_levels):
        adjusted_beta = beta * (1 - level)
        results = odeint(deriv, initial_conditions, t, args=(N, adjusted_beta, gamma, mu, W, a))
        total_infected = results[:, 1] + results[:, 5] + results[:, 9]
        
        # Find local maxima
        maxima_indices, _ = find_local_maxima(total_infected)
        
        # Default placeholders
        peak_reduction_str = "None"
        peak_delay_str = "None"
        
        if len(maxima_indices) > 0:
            idx = maxima_indices[0]
            peak_current = total_infected[idx]
            peak_day_current = t[idx]
            
            # Compute peak reduction
            if peak_base > 0:
                peak_reduction_val = ((peak_base - peak_current) / peak_base) * 100
                peak_reduction_str = f"{peak_reduction_val:.2f}%"
            else:
                peak_reduction_str = "None"
            
            # Compute delay
            peak_delay_val = peak_day_current - peak_time_base
            if peak_delay_val >= 0:
                peak_delay_str = f"{peak_delay_val:.2f} days"
            else:
                peak_delay_str = "None"
        
        label_str = (f"Level: {int(level * 100)}% reduction "
                     f"({1 - level:.2f} multiplier), "
                     f"Peak Red: {peak_reduction_str}, "
                     f"Delay: {peak_delay_str}")
        
        plt.plot(t, total_infected, color=colors[i], label=label_str)
        plt.scatter(t[maxima_indices], total_infected[maxima_indices], color='black', zorder=5)
    
    plt.title('Total Infectious Population under Population-Wide Quarantine')
    plt.xlabel('Days')
    plt.ylabel('Total Infectious Population (All Groups)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig("Figures/population_quarantine_Total_Infected_Population.png")
    plt.show()

# --- PLOTTING CALLS ---
plot_group_infected_4_levels_population(0, "Group 1 (Children)",
                                        beta.copy(), N, gamma, mu, W, a,
                                        initial_conditions, time_span)

plot_group_infected_4_levels_population(4, "Group 2 (Adults)",
                                        beta.copy(), N, gamma, mu, W, a,
                                        initial_conditions, time_span)

plot_group_infected_4_levels_population(8, "Group 3 (Seniors)",
                                        beta.copy(), N, gamma, mu, W, a,
                                        initial_conditions, time_span)

plot_total_infected_4_levels_population(beta.copy(), N, gamma, mu, W, a,
                                        initial_conditions, time_span)


# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 21:17:48 2025

@author: joonc

This version applies a % reduction (multiplier = ) to each individual element 
of the 3x3 beta transmission matrix one at a time and records the total infected peak.
A 3x3 subplot figure is generated so that each cell shows the simulation result 
(for total infected population) when a specific beta element is reduced.
"""
import os
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

# ===================== Font-size patch (추가) =====================
plt.rcParams.update({
    'font.size': 20,        # 기본 폰트
    'axes.titlesize': 20,   # 서브플롯 제목
    'axes.labelsize': 15,   # x/y 라벨
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 15,    # 범례
    'figure.titlesize': 30,
})
# ================================================================


# Create "Figures" directory if it doesn't exist
if not os.path.exists("Figures"):
    os.makedirs("Figures")

# Define a single quarantine multiplier (here, 0.3 reduction => multiplier of 0.7)
reduction_factor = 0.2  # Change this value as needed

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

# Set the baseline beta matrix (we will modify copies of this for each scenario)
beta = transmission_rates

# Define initial conditions for the three groups (including vaccinated compartments)
initial_conditions = [
    S_init[0], I_init[0], R_init[0], V_init[0],  # Group 1 (Children)
    S_init[1], I_init[1], R_init[1], V_init[1],  # Group 2 (Adults)
    S_init[2], I_init[2], R_init[2], V_init[2]   # Group 3 (Seniors)
]

# SIRV model differential equations including vaccinated compartments
def deriv(y, t, N, beta, gamma, mu, W, vaccination_rates):
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y
    
    # Force of infection (lambda_i) for each group
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

# Function to find the local maxima in a time series
def find_local_maxima(I):
    maxima_indices = argrelextrema(I, np.greater)[0]
    maxima_values = I[maxima_indices]
    return maxima_indices, maxima_values

# Function to simulate the model with a modified beta (one element reduced)
def simulate_modified_beta(modified_i, modified_j):
    t = np.linspace(0, time_span, int(time_span))
    beta_modified = beta.copy()
    # Apply reduction to the specified beta element (one at a time)
    beta_modified[modified_i, modified_j] *= reduction_factor
    sol = odeint(deriv, initial_conditions, t, args=(N, beta_modified, gamma, mu, W, a))
    # Total infected population = I1 + I2 + I3
    total_infected = sol[:, 1] + sol[:, 5] + sol[:, 9]
    peak_val = np.max(total_infected)
    peak_time = t[np.argmax(total_infected)]
    return t, total_infected, peak_val, peak_time

# Optional: Compute the baseline (unmodified) simulation for comparison
t = np.linspace(0, time_span, int(time_span))
sol_baseline = odeint(deriv, initial_conditions, t, args=(N, beta, gamma, mu, W, a))
baseline_total = sol_baseline[:, 1] + sol_baseline[:, 5] + sol_baseline[:, 9]
baseline_peak = np.max(baseline_total)
baseline_peak_time = t[np.argmax(baseline_total)]

# Function to create a 3x3 grid plot comparing all 9 modified scenarios
def plot_component_wise_beta_reduction():
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
    # Dictionary to record peak information for each scenario
    peak_info = {}
    
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            t, total_infected, peak_val, peak_time = simulate_modified_beta(i, j)
            # Record the peak info
            key = f'beta[{i}][{j}]'
            peak_info[key] = {'peak_value': peak_val, 'peak_time': peak_time}
            
            # Plot the modified scenario curve
            ax.plot(t, total_infected, label=f'Modified {key}', color='blue')
            ax.scatter(peak_time, peak_val, color='red', zorder=5, label=f'Peak: {peak_val:.2f} at t={peak_time:.1f}')
            
            # Plot the baseline curve for comparison (dashed gray)
            ax.plot(t, baseline_total, linestyle='--', color='gray', label='Baseline')
            
            ax.set_title(f"Reduction on {key}")
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("Total Infected")
            ax.legend(fontsize=10)
    
    fig.suptitle(f"Total Infected Population with Component-wise {reduction_factor} β multipliers on the Matrix", fontsize=20, fontweight='normal')
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    # Save the figure with the reduction factor included in the file name
    filename = f"Figures/component_wise_beta_reduction_multiplier_{reduction_factor}.png"
    plt.savefig(filename)
    plt.show()
    
    # Print the recorded peak information for each scenario
    for key, info in peak_info.items():
        print(f"{key}: Peak Infected = {info['peak_value']:.2f} at t = {info['peak_time']:.2f} days")

# Execute the plotting function
plot_component_wise_beta_reduction()
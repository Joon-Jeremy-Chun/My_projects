# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 22:10:55 2025

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
    infectious_init = [1, 1, 1]  # Define Infectious_init
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

# SIRV model differential equations
def deriv(y, t, N, beta, gamma, mu, W, vaccination_rates):
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y
    
    lambda1 = beta[0, 0] * I1/N[0] + beta[0, 1] * I2/N[1] + beta[0, 2] * I3/N[2]
    lambda2 = beta[1, 0] * I1/N[0] + beta[1, 1] * I2/N[1] + beta[1, 2] * I3/N[2]
    lambda3 = beta[2, 0] * I1/N[0] + beta[2, 1] * I2/N[1] + beta[2, 2] * I3/N[2]
    
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

# Function to simulate with reduction logic
def simulate_with_reduction(beta, initial_conditions, N, gamma, mu, W, a, time_span, k, r_sub):
    t1 = np.linspace(0, k, k + 1)  # First k days
    t2 = np.linspace(k, time_span, int(time_span - k) + 1)  # From k+1 onward
    
    results_t1 = odeint(deriv, initial_conditions, t1, args=(N, beta, gamma, mu, W, a))
    initial_conditions_t2 = results_t1[-1, :]  # Last state from t1
    
    reduced_beta = beta * r_sub  # Uniform reduction
    results_t2 = odeint(deriv, initial_conditions_t2, t2, args=(N, reduced_beta, gamma, mu, W, a))
    
    t_combined = np.concatenate((t1, t2[1:]))  # Avoid duplicate time at t = k
    results_combined = np.vstack((results_t1, results_t2[1:, :]))  # Combine both simulations
    
    return t_combined, results_combined

# Simulation parameters
k = 30  # Days before applying reduction
r_sub = 0.25  # Reduction factor (1-x% reduction in beta)

# Run simulation
t_combined, results_combined = simulate_with_reduction(beta, initial_conditions, N, gamma, mu, W, a, time_span, k, r_sub)

# Extract results for infected groups
I_total_combined = results_combined[:, 1] + results_combined[:, 5] + results_combined[:, 9]
I_groups_combined = {
    "Group 1 (Children)": results_combined[:, 1],
    "Group 2 (Adults)": results_combined[:, 5],
    "Group 3 (Seniors)": results_combined[:, 9]
}

# Plot total infected population
plt.figure(figsize=(10, 6))
plt.plot(t_combined, I_total_combined, label="Total Infected Population")
plt.axvline(x=k, color="red", linestyle="--", label=f"Policy Change at Day {k}")
plt.title("Total Infected Population with Policy Change")
plt.xlabel("Days")
plt.ylabel("Infectious Population")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Plot for each group
colors = ["blue", "green", "orange"]
for idx, (label, I) in enumerate(I_groups_combined.items()):
    plt.figure(figsize=(10, 6))
    plt.plot(t_combined, I, label=label, color=colors[idx])
    plt.axvline(x=k, color="red", linestyle="--", label=f"Policy Change at Day {k}")
    plt.title(f"Infectious Population for {label} with Policy Change")
    plt.xlabel("Days")
    plt.ylabel("Infectious Population")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
#%%

# Function to compute relative maxima for all groups and total population
def compute_relative_maxima(t, results_combined):
    maxima_results = {}
    # Individual groups
    for idx, label in enumerate(["Group 1 (Children)", "Group 2 (Adults)", "Group 3 (Seniors)"]):
        # Extract infected population for the group
        I_group = results_combined[:, idx * 4 + 1]
        
        # Find relative maxima
        maxima_indices = argrelextrema(I_group, np.greater)[0]
        maxima_values = I_group[maxima_indices]
        maxima_days = t[maxima_indices]
        
        # Store the results
        maxima_results[label] = {
            "Maxima Values": maxima_values,
            "Days After Start": maxima_days
        }
    
    # Total population
    I_total = results_combined[:, 1] + results_combined[:, 5] + results_combined[:, 9]
    maxima_indices_total = argrelextrema(I_total, np.greater)[0]
    maxima_values_total = I_total[maxima_indices_total]
    maxima_days_total = t[maxima_indices_total]
    
    # Store the results for the total population
    maxima_results["Total Population"] = {
        "Maxima Values": maxima_values_total,
        "Days After Start": maxima_days_total
    }
    
    return maxima_results

# Compute relative maxima for all groups and total population
relative_maxima_results = compute_relative_maxima(t_combined, results_combined)

# Print results
for group, data in relative_maxima_results.items():
    print(f"\n{group}:")
    for i, (value, day) in enumerate(zip(data["Maxima Values"], data["Days After Start"])):
        print(f"  Relative Maximum {i + 1}: {value:.2f} (Day {day:.2f})")

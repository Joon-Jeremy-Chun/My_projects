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
            time_span, population_size, susceptible_init,
            infectious_init, recovered_init, vaccinated_init)

# Load the parameters from the Excel file
file_path = 'Inputs.xlsx'
(gamma, mu, W, time_span, N, S_init, I_init, R_init, V_init) = load_parameters(file_path)

# Load the transmission_rates matrix from the CSV file
transmission_rates_csv_path = 'DataSets/Fitted_Beta_Matrix.csv'
transmission_rates = pd.read_csv(transmission_rates_csv_path).iloc[0:3, 1:4].values

# Override the beta parameter with the loaded transmission_rates
beta = transmission_rates

# Define custom vaccination rates directly in the script
#vaccination_rates = [0.1, 0.1, 0.1]
vaccination_rates = [0.5, 0.5, 0.5]  
  

# Define initial conditions for the three groups (including vaccinated compartments)
initial_conditions = [
    S_init[0], I_init[0], R_init[0], V_init[0],  # S1, I1, R1, V1 (Group 1)
    S_init[1], I_init[1], R_init[1], V_init[1],  # S2, I2, R2, V2 (Group 2)
    S_init[2], I_init[2], R_init[2], V_init[2]   # S3, I3, R3, V3 (Group 3)
]

# SIRV model differential equations including vaccinated compartments
def deriv(y, t, N, beta, gamma, mu, W, vaccination_rates):
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y
    
    # Force of infection (\u03bb_i) for each group
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

# Function to plot the total infected population over time
def plot_total_infected_with_vaccination(beta, N, gamma, mu, W, vaccination_rates, initial_conditions, time_span):
    t = np.linspace(0, time_span, int(time_span))
    plt.figure(figsize=(10, 6))
    
    # Simulate Level 0 (no quarantine)
    results_base = odeint(deriv, initial_conditions, t, args=(N, beta, gamma, mu, W, vaccination_rates))
    total_base = results_base[:, 1] + results_base[:, 5] + results_base[:, 9]
    peak_base = np.max(total_base)
    peak_time_base = t[np.argmax(total_base)]
    
    # Quarantine levels and colors
    quarantine_levels = [0, 0.1]  #Here Quarantine 
    colors = plt.cm.rainbow(np.linspace(0, 1, len(quarantine_levels)))
    
    for i, level in enumerate(quarantine_levels):
        adjusted_beta = beta.copy() * (1 - level)
        results = odeint(deriv, initial_conditions, t, args=(N, adjusted_beta, gamma, mu, W, vaccination_rates))
        total_infected = results[:, 1] + results[:, 5] + results[:, 9]
        
        plt.plot(t, total_infected, color=colors[i], label=f'Level: {int(level * 100)}%, Peak Reduction: {((peak_base - np.max(total_infected)) / peak_base) * 100:.2f}%, Delay: {t[np.argmax(total_infected)] - peak_time_base:.2f} days')
    
    plt.title('Total Infectious Population with Vaccination under Quarantine Levels')
    plt.xlabel('Days')
    plt.ylabel('Total Infectious Population (All Groups)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# Plot the total infected population
plot_total_infected_with_vaccination(beta.copy(), N, gamma, mu, W, vaccination_rates, initial_conditions, time_span)

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:25:59 2025

@author: joonc

Extended Model with Dynamic Vaccination Strategy (3.5)
"""

import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# Function to load model parameters from Excel
def load_parameters(file_path):
    df = pd.read_excel(file_path, header=None)
    
    recovery_rates = df.iloc[4, 0:3].values
    maturity_rates = df.iloc[6, 0:2].values
    waning_immunity_rate = df.iloc[8, 0]
    time_span = df.iloc[12, 0]
    population_size = df.iloc[14, 0:3].values
    susceptible_init = df.iloc[14, 0:3].values
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
beta = pd.read_csv(transmission_rates_csv_path).iloc[0:3, 1:4].values

# Define initial conditions
initial_conditions = [
    S_init[0], I_init[0], R_init[0], V_init[0],  
    S_init[1], I_init[1], R_init[1], V_init[1],  
    S_init[2], I_init[2], R_init[2], V_init[2]   
]

# Option 1: Manually defined vaccination rates
vaccination_rates = np.array([0.02, 0.02, 0.02])  # Initial rate

# Option 2: Define total ratio (k) and days (d) for dynamic vaccination rate calculation
k1, d1 = 0.8, 30  # 80% in 30 days for Children
k2, d2 = 0.7, 45  # 70% in 45 days for Adults
k3, d3 = 0.9, 60  # 90% in 60 days for Seniors

# Calculate daily vaccination rate for each group
x1 = k1 ** (1/d1)
x2 = k2 ** (1/d2)
x3 = k3 ** (1/d3)

print(f"Daily Vaccination Rate for Children (x1): {x1:.5f}")
print(f"Daily Vaccination Rate for Adults (x2): {x2:.5f}")
print(f"Daily Vaccination Rate for Seniors (x3): {x3:.5f}")

# Use calculated values as vaccination rates
vaccination_rates_dynamic = np.array([x1, x2, x3])

# Define three stage dates and % of boosts
t1, t2 = 30, 60
B_timeP0, B_timeP1, B_timeP2 = 1.0, 1.0, 1.0

# Dynamic vaccination function
def vaccination_strategy(t, use_dynamic=True):
    """
    Defines time-dependent vaccination rates.
    """
    rates = vaccination_rates_dynamic if use_dynamic else vaccination_rates
    if t < t1:
        return rates * B_timeP0  # Initial phase
    elif t1 <= t < t2:
        return rates * B_timeP1  # Full-scale phase
    else:
        return rates * B_timeP2  # Booster phase

# Extended SIRV model with time-dependent vaccination
def deriv(y, t, N, beta, gamma, mu, W):
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y
    a_t = vaccination_strategy(t)
    
    lambda1 = beta[0, 0] * I1/N[0] + beta[0, 1] * I2/N[1] + beta[0, 2] * I3/N[2]
    lambda2 = beta[1, 0] * I1/N[0] + beta[1, 1] * I2/N[1] + beta[1, 2] * I3/N[2]
    lambda3 = beta[2, 0] * I1/N[0] + beta[2, 1] * I2/N[1] + beta[2, 2] * I3/N[2]
    
    dS1dt = -lambda1 * S1 - a_t[0] * S1 + W * R1 + W * V1
    dI1dt = lambda1 * S1 - gamma[0] * I1 - mu[0] * I1
    dR1dt = gamma[0] * I1 - W * R1 - mu[0] * R1
    dV1dt = a_t[0] * S1 - W * V1
    
    dS2dt = -lambda2 * S2 + mu[0] * S1 - a_t[1] * S2 + W * R2 + W * V2
    dI2dt = lambda2 * S2 + mu[0] * I1 - gamma[1] * I2 - mu[1] * I2
    dR2dt = gamma[1] * I2 + mu[0] * R1 - W * R2 - mu[1] * R2
    dV2dt = a_t[1] * S2 - W * V2
    
    dS3dt = -lambda3 * S3 + mu[1] * S2 - a_t[2] * S3 + W * R3 + W * V3
    dI3dt = lambda3 * S3 + mu[1] * I2 - gamma[2] * I3
    dR3dt = gamma[2] * I3 + mu[1] * R2 - W * R3
    dV3dt = a_t[2] * S3 - W * V3
    
    return [dS1dt, dI1dt, dR1dt, dV1dt,
            dS2dt, dI2dt, dR2dt, dV2dt,
            dS3dt, dI3dt, dR3dt, dV3dt]

# Time span
t = np.linspace(0, time_span, int(time_span))

# Solve the model
dyn_results = odeint(deriv, initial_conditions, t, args=(N, beta, gamma, mu, W))

# Computation of peak values and corresponding days
I_children = dyn_results[:, 1]
I_adults = dyn_results[:, 5]
I_seniors = dyn_results[:, 9]
I_total = I_children + I_adults + I_seniors

peak_idx_children = np.argmax(I_children)
peak_idx_adults = np.argmax(I_adults)
peak_idx_seniors = np.argmax(I_seniors)
peak_idx_total = np.argmax(I_total)

print("Peak Infections and Days:")
print(f"Children: Peak = {I_children[peak_idx_children]:.2f}, Day = {t[peak_idx_children]:.2f}")
print(f"Adults: Peak = {I_adults[peak_idx_adults]:.2f}, Day = {t[peak_idx_adults]:.2f}")
print(f"Seniors: Peak = {I_seniors[peak_idx_seniors]:.2f}, Day = {t[peak_idx_seniors]:.2f}")
print(f"Total Population: Peak = {I_total[peak_idx_total]:.2f}, Day = {t[peak_idx_total]:.2f}")

# Create a time array to observe vaccination changes
t_plot = np.linspace(0, time_span, int(time_span))

# Compute vaccination rates over time
vaccination_rates_over_time = np.array([vaccination_strategy(time) for time in t_plot])

# Convert to separate group rates for plotting
vaccination_children = vaccination_rates_over_time[:, 0]
vaccination_adults = vaccination_rates_over_time[:, 1]
vaccination_seniors = vaccination_rates_over_time[:, 2]

# Plot vaccination rates over time
plt.figure(figsize=(10, 6))
plt.plot(t_plot, vaccination_children, label="Children", color='blue')
plt.plot(t_plot, vaccination_adults, label="Adults", color='green')
plt.plot(t_plot, vaccination_seniors, label="Seniors", color='red')
plt.title("Vaccination Strategy Over Time")
plt.xlabel("Days")
plt.ylabel("Vaccination Rate")
plt.legend()
plt.grid()
plt.show()

# Plot individual group figures with maxima
fig_labels = ['Group 1 (Children)', 'Group 2 (Adults)', 'Group 3 (Seniors)', 'Total Infected']
for i, label in enumerate(fig_labels[:-1]):
    plt.figure(figsize=(10, 6))
    I = dyn_results[:, 1 + i * 4]
    peak_idx = np.argmax(I)
    plt.plot(t, I, label=f'Infected {label}')
    plt.scatter(t[peak_idx], I[peak_idx], color='red', marker='o', label=f'Max: ({t[peak_idx]:.2f}, {I[peak_idx]:.2f})')
    plt.title(f'Infectious Population for {label} with Dynamic Vaccination')
    plt.xlabel('Days')
    plt.ylabel('Number of Infected Individuals')
    plt.legend()
    plt.grid()
    plt.show()

plt.figure(figsize=(10, 6))
I_total = dyn_results[:, 1] + dyn_results[:, 5] + dyn_results[:, 9]
peak_idx = np.argmax(I_total)
plt.plot(t, I_total, label='Total Infected')
plt.scatter(t[peak_idx], I_total[peak_idx], color='red', marker='o', label=f'Max: ({t[peak_idx]:.2f}, {I_total[peak_idx]:.2f})')
plt.title('Total Infectious Population with Dynamic Vaccination')
plt.xlabel('Days')
plt.ylabel('Number of Infected Individuals')
plt.legend()
plt.grid()
plt.show()

# Plot all groups in one figure with maxima
plt.figure(figsize=(10, 6))
I_children = dyn_results[:, 1]
I_adults = dyn_results[:, 5]
I_seniors = dyn_results[:, 9]
peak_idx_children = np.argmax(I_children)
peak_idx_adults = np.argmax(I_adults)
peak_idx_seniors = np.argmax(I_seniors)
plt.plot(t, I_children, label='Children', color='blue')
plt.scatter(t[peak_idx_children], I_children[peak_idx_children], color='red', marker='o', label=f'Max (Children): ({t[peak_idx_children]:.2f}, {I_children[peak_idx_children]:.2f})')
plt.plot(t, I_adults, label='Adults', color='green')
plt.scatter(t[peak_idx_adults], I_adults[peak_idx_adults], color='red', marker='o', label=f'Max (Adults): ({t[peak_idx_adults]:.2f}, {I_adults[peak_idx_adults]:.2f})')
plt.plot(t, I_seniors, label='Seniors', color='red')
plt.scatter(t[peak_idx_seniors], I_seniors[peak_idx_seniors], color='red', marker='o', label=f'Max (Seniors): ({t[peak_idx_seniors]:.2f}, {I_seniors[peak_idx_seniors]:.2f})')
plt.title('Comparison of Infectious Populations Across Groups')
plt.xlabel('Days')
plt.ylabel('Number of Infected Individuals')
plt.legend()
plt.grid()
plt.show()

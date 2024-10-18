# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 20:18:08 2024

@author: joonc
"""
import pandas as pd
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# Function to load model parameters and initial conditions from Excel
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
    
    # Initial conditions: Susceptible, Infectious, Recovered, Vaccinated (each for 3 groups)
    susceptible_init = df.iloc[14, 0:3].values  # Define population size
    infectious_init = df.iloc[16, 0:3].values  
    recovered_init = df.iloc[18, 0:3].values   
    vaccinated_init = df.iloc[20, 0:3].values  
    
    # Quarantine-related data
    under_quarantine_transmission_rates = df.iloc[22:26, 0:3].values
    quarantine_day = df.iloc[26, 0]
    
    return (transmission_rates, recovery_rates, maturity_rates, waning_immunity_rate,
            vaccination_rates, time_span, population_size, susceptible_init, infectious_init,
            recovered_init, vaccinated_init, under_quarantine_transmission_rates, quarantine_day)

# Function to load observed data from CSV
def load_observed_data(file_path, start_date, end_date):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    filtered_data = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    return filtered_data

# SIRV model differential equations including quarantine dynamics
def deriv(y, t, N, beta, gamma, mu, W, a):
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y
    
    # Force of infection (lambda_i) for each group
    lambda1 = beta[0, 0] * I1 / N[0] + beta[0, 1] * I2 / N[1] + beta[0, 2] * I3 / N[2]
    lambda2 = beta[1, 0] * I1 / N[0] + beta[1, 1] * I2 / N[1] + beta[1, 2] * I3 / N[2]
    lambda3 = beta[2, 0] * I1 / N[0] + beta[2, 1] * I2 / N[1] + beta[2, 2] * I3 / N[2]
    
    # Differential equations for each compartment
    # Group 1 (e.g., children)
    dS1dt = -lambda1 * S1 - a[0] * S1 + W * R1 + W * V1
    dI1dt = lambda1 * S1 - gamma[0] * I1 - mu[0] * I1
    dR1dt = gamma[0] * I1 - W * R1 - mu[0] * R1
    dV1dt = a[0] * S1 - W * V1
    
    # Group 2 (e.g., adults)
    dS2dt = -lambda2 * S2 + mu[0] * S1 - a[1] * S2 + W * R2 + W * V2
    dI2dt = lambda2 * S2 + mu[0] * I1 - gamma[1] * I2 - mu[1] * I2
    dR2dt = gamma[1] * I2 + mu[0] * R1 - W * R2 - mu[1] * R2
    dV2dt = a[1] * S2 - W * V2
    
    # Group 3 (e.g., seniors)
    dS3dt = -lambda3 * S3 + mu[1] * S2 - a[2] * S3 + W * R3 + W * V3
    dI3dt = lambda3 * S3 + mu[1] * I2 - gamma[2] * I3
    dR3dt = gamma[2] * I3 + mu[1] * R2 - W * R3
    dV3dt = a[2] * S3 - W * V3
    
    return dS1dt, dI1dt, dR1dt, dV1dt, dS2dt, dI2dt, dR2dt, dV2dt, dS3dt, dI3dt, dR3dt, dV3dt

# Objective function to fit transmission rates
def objective(params, t, N, initial_conditions, I_data, gamma, mu, W, a):
    # Update the transmission rates based on input params
    beta_flat = params[:9]
    beta = beta_flat.reshape((3, 3))
    
    # Solve the system with these parameters
    results = odeint(deriv, initial_conditions, t, args=(N, beta, gamma, mu, W, a))
    _, I1, _, _, _, I2, _, _, _, I3, _, _ = results.T
    
    # Calculate the sum of squared errors between model and observed data
    error = np.sum((I1 - I_data[0])**2) + np.sum((I2 - I_data[1])**2) + np.sum((I3 - I_data[2])**2)
    return error

# Load parameters and observed data
inputs_path = 'Inputs.xlsx'
data_path = 'DataSets/Korea_threeGroups_covid19_data.csv'
(beta, gamma, mu, W, a, time_span, N, S_init, I_init, R_init, V_init, beta_quarantine, quarantine_day) = load_parameters(inputs_path)
observed_data = load_observed_data(data_path, "2020-03-01", "2020-04-01")

# Extract observed infected and susceptible data for each group
I1_data = observed_data[observed_data['new_age_group'] == '0-19']['new_confirmed_cases'].rolling(14).sum().dropna().values
I2_data = observed_data[observed_data['new_age_group'] == '20-49']['new_confirmed_cases'].rolling(14).sum().dropna().values
I3_data = observed_data[observed_data['new_age_group'] == '50-80+']['new_confirmed_cases'].rolling(14).sum().dropna().values
I_data = [I1_data, I2_data, I3_data]

# Initial conditions for the three groups
initial_conditions = [
    S_init[0], I_init[0], R_init[0], V_init[0],  # Group 1
    S_init[1], I_init[1], R_init[1], V_init[1],  # Group 2
    S_init[2], I_init[2], R_init[2], V_init[2]   # Group 3
]

# Time grid for fitting
t = np.linspace(0, len(I1_data) - 1, len(I1_data))

# Initial guesses for the parameters to optimize
initial_guess = beta.flatten()

# Optimize transmission rates
result = minimize(objective, initial_guess, args=(t, N, initial_conditions, I_data, gamma, mu, W, a))

# Extract fitted transmission rates
fitted_beta = result.x.reshape((3, 3))
print("Fitted Transmission Rates:")
print(fitted_beta)

# Plot results
results = odeint(deriv, initial_conditions, t, args=(N, fitted_beta, gamma, mu, W, a))
_, I1, _, _, _, I2, _, _, _, I3, _, _ = results.T

# Plot the observed vs. fitted infected data for each group
plt.figure(figsize=(10, 6))
plt.plot(t, I1_data, 'ro', label='Observed Infected (0-19, 14-day rolling)')
plt.plot(t, I1, 'r-', label='Fitted Infected (0-19)')
plt.plot(t, I2_data, 'bo', label='Observed Infected (20-49, 14-day rolling)')
plt.plot(t, I2, 'b-', label='Fitted Infected (20-49)')
plt.plot(t, I3_data, 'go', label='Observed Infected (50-80+, 14-day rolling)')
plt.plot(t, I3, 'g-', label='Fitted Infected (50-80+)')
plt.xlabel('Days')
plt.ylabel('Infected Population (14-day rolling)')
plt.legend()
plt.title('Parameter Fitting for Transmission Rates')
plt.show()

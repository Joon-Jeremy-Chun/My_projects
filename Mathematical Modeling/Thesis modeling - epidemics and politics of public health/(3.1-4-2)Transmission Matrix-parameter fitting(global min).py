# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 07:45:48 2024

@author: joonc
"""

import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution

# Step 1: Load Real Data and Initial Parameters
file_path_data = 'DataSets/Korea_threeGroups_covid19_data.csv'
file_path_inputs = 'Inputs.xlsx'

# Load the dataset
data = pd.read_csv(file_path_data)

# Ensure proper date parsing
data['date'] = pd.to_datetime(data['date'])

# Load initial parameters from inputs
inputs = pd.read_excel(file_path_inputs, header=None)

# Extract initial populations, recovery rates, vaccination rates, and waning immunity rates
initial_populations = inputs.iloc[14, 0:3].values
recovery_rates = inputs.iloc[4, 0:3].values
vaccination_rates = inputs.iloc[10, 0:3].values
waning_immunity_rate = inputs.iloc[8, 0]  # Single value

# Define compartments and initialize
age_groups = ['0-19', '20-49', '50-80+']
S, I, R, V = {}, {}, {}, {}
for idx, age_group in enumerate(age_groups):
    S[age_group] = [initial_populations[idx]]  # Initialize Susceptible
    I[age_group] = [0]  # Initialize Infectious
    R[age_group] = [0]  # Initialize Recovered
    V[age_group] = [0]  # Initialize Vaccinated

# Time steps (days) based on the dataset
dates = pd.to_datetime(data['date'].unique())
time_steps = len(dates)

# Update compartments daily based on ODEs
for t in range(1, time_steps):
    current_date = dates[t]
    daily_data = data[data['date'] == current_date]

    for idx, age_group in enumerate(age_groups):
        # New cases directly from dataset
        filtered_data = daily_data[daily_data['new_age_group'] == age_group]
        new_cases = filtered_data['new_confirmed_cases'].sum()

        # Parameters for the current age group
        gamma = recovery_rates[idx]
        a = vaccination_rates[idx]
        delta = waning_immunity_rate

        # Update ODEs compartments
        S_new = S[age_group][-1] - new_cases - a * S[age_group][-1] + delta * (R[age_group][-1] + V[age_group][-1])
        I_new = I[age_group][-1] + new_cases - gamma * I[age_group][-1]
        R_new = R[age_group][-1] + gamma * I[age_group][-1] - delta * R[age_group][-1]
        V_new = V[age_group][-1] + a * S[age_group][-1] - delta * V[age_group][-1]

        # Ensure updates use real dataset for infectious compartment
        I_new = max(new_cases, I_new)

        # Append updated values ensuring non-negativity
        S[age_group].append(max(S_new, 0))
        I[age_group].append(max(I_new, 0))
        R[age_group].append(max(R_new, 0))
        V[age_group].append(max(V_new, 0))

# Create a DataFrame to store results
results = pd.DataFrame(index=dates)
for age_group in age_groups:
    results[f"S_{age_group}"] = S[age_group]
    results[f"I_{age_group}"] = I[age_group]
    results[f"R_{age_group}"] = R[age_group]
    results[f"V_{age_group}"] = V[age_group]

# Calculate total population compartments
results["S_total"] = results[[f"S_{age}" for age in age_groups]].sum(axis=1)
results["I_total"] = results[[f"I_{age}" for age in age_groups]].sum(axis=1)
results["R_total"] = results[[f"R_{age}" for age in age_groups]].sum(axis=1)
results["V_total"] = results[[f"V_{age}" for age in age_groups]].sum(axis=1)

def loss_function(beta_values, age_groups, start_date, end_date):
    beta_matrix = beta_values.reshape(3, 3)  # Reshape to 3x3 matrix
    start_idx = results.index.get_loc(start_date)
    end_idx = results.index.get_loc(end_date)

    simulated_I = []
    observed_I = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

    for t in range(start_idx, end_idx + 1):
        new_I = []
        for i, age_i in enumerate(age_groups):
            # The force of infection lambda_i
            lambda_i = 0
            for j in range(3):
                # Example: Transmission depends on current infectious counts
                lambda_i += beta_matrix[i, j] * I[age_groups[j]][t]
            new_I.append(lambda_i)
        simulated_I.append(new_I)

    simulated_I = pd.DataFrame(simulated_I, columns=age_groups, index=results.index[start_idx:end_idx + 1])
    observed_I = observed_I.pivot(index='date', columns='new_age_group', values='new_confirmed_cases')

    # Align the indices
    simulated_I = simulated_I.loc[observed_I.index]

    # Mean squared error for all groups
    mse = ((simulated_I.values - observed_I.values) ** 2).mean()
    return mse

# Define fitting period (adjust as needed)
start_fit_date = pd.to_datetime('2020-01-20')
end_fit_date = pd.to_datetime('2020-03-20')

# Define bounds for the beta parameters (3x3 matrix)
bounds = [(0, 10)] * 9

# Perform global optimization using Differential Evolution
result = differential_evolution(loss_function, bounds=bounds, args=(age_groups, start_fit_date, end_fit_date), maxiter=1000, polish=True)

# Extract optimized beta matrix
optimized_beta_matrix = result.x.reshape(3, 3)
print("Optimized Beta Matrix (Global):")
print(optimized_beta_matrix)

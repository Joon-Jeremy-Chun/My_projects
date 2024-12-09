# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 06:41:01 2024

@author: joonc
"""

import pandas as pd
import matplotlib.pyplot as plt

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

        # Append updated values
        S[age_group].append(max(S_new, 0))  # Ensure non-negative values
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

# Plotting the results
plt.figure(figsize=(12, 8))

# Plot total compartments
plt.plot(results.index, results["S_total"], label="Susceptible (Total)")
plt.plot(results.index, results["I_total"], label="Infectious (Total)")
plt.plot(results.index, results["R_total"], label="Recovered (Total)")
plt.plot(results.index, results["V_total"], label="Vaccinated (Total)")

# Add labels and legend
plt.xlabel("Date")
plt.ylabel("Population")
plt.title("SIRV Model Compartments Over Time (Total)")
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()

# Plot compartments for each age group
for age_group in age_groups:
    plt.figure(figsize=(12, 8))
    plt.plot(results.index, results[f"S_{age_group}"], label=f"Susceptible ({age_group})")
    plt.plot(results.index, results[f"I_{age_group}"], label=f"Infectious ({age_group})")
    plt.plot(results.index, results[f"R_{age_group}"], label=f"Recovered ({age_group})")
    plt.plot(results.index, results[f"V_{age_group}"], label=f"Vaccinated ({age_group})")

    # Add labels and legend
    plt.xlabel("Date")
    plt.ylabel("Population")
    plt.title(f"SIRV Model Compartments Over Time ({age_group})")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()








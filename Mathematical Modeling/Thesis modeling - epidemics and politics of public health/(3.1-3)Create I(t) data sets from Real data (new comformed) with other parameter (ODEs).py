# -*- coding: utf-8 -*-
"""
Created on Mon Dec  16 10:50:30 2024
@author: joonc
"""

import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load Data and Parameters
file_path_data = 'DataSets/Korea_threeGroups_covid19_data.csv'
file_path_inputs = 'Inputs.xlsx'
output_file_path = 'DataSets/Korea_threeGroups_SIRV_parameter.csv'

# Load dataset
data = pd.read_csv(file_path_data)
data['date'] = pd.to_datetime(data['date'])


# Load parameters
inputs = pd.read_excel(file_path_inputs, header=None)
initial_populations = inputs.iloc[14, 0:3].values
recovery_rates = inputs.iloc[4, 0:3].values
vaccination_rates = inputs.iloc[10, 0:3].values
waning_immunity_rate = inputs.iloc[8, 0]

# Define age groups and compartments
age_groups = ['0-19', '20-49', '50-80+']
compartments = ['S', 'I', 'R', 'V']

# Initialize compartments
S, I, R, V = {}, {}, {}, {}
for idx, age_group in enumerate(age_groups):
    S[age_group] = [initial_populations[idx]]  # Susceptible
    I[age_group] = [0]  # Infectious
    R[age_group] = [0]  # Recovered
    V[age_group] = [0]  # Vaccinated

# Time steps and dates
dates = pd.to_datetime(data['date'].unique())
time_steps = len(dates)

# Step 2: Update Compartments Daily
for t in range(time_steps):
    current_date = dates[t]
    daily_data = data[data['date'] == current_date]
    
    for idx, age_group in enumerate(age_groups):
        # New cases
        filtered_data = daily_data[daily_data['new_age_group'] == age_group]
        new_cases = filtered_data['new_confirmed_cases'].sum()

        # Parameters for the current age group
        gamma = recovery_rates[idx]
        a = vaccination_rates[idx]
        delta = waning_immunity_rate

        # Initial day update: Apply new cases directly to I and adjust S
        if t == 0:
            S[age_group][0] -= new_cases  # Deduct new cases from susceptible
            I[age_group][0] += new_cases  # Add new cases to infectious

        else:
            # Previous day's compartments
            S_prev = S[age_group][-1]
            I_prev = I[age_group][-1]
            R_prev = R[age_group][-1]
            V_prev = V[age_group][-1]

            # Update compartments
            S_new = S_prev - new_cases - a * S_prev + delta * (R_prev + V_prev)
            I_new = I_prev + new_cases - gamma * I_prev
            R_new = R_prev + gamma * I_prev - delta * R_prev
            V_new = V_prev + a * S_prev - delta * V_prev

            # Append new values (ensure non-negative)
            S[age_group].append(max(S_new, 0))
            I[age_group].append(max(I_new, 0))
            R[age_group].append(max(R_new, 0))
            V[age_group].append(max(V_new, 0))


# Step 3: Combine Results into a Single DataFrame
combined_results = pd.DataFrame({'Date': dates})

for age_group in age_groups:
    combined_results[f"S_{age_group}"] = S[age_group]
    combined_results[f"I_{age_group}"] = I[age_group]
    combined_results[f"R_{age_group}"] = R[age_group]
    combined_results[f"V_{age_group}"] = V[age_group]

# Save to the specified CSV file
combined_results.to_csv(output_file_path, index=False)
print(f"Results saved to: {output_file_path}")

# Step 4: Plot Results for Each Age Group
for age_group in age_groups:
    plt.figure(figsize=(10, 6))
    plt.plot(combined_results['Date'], combined_results[f"S_{age_group}"], label="Susceptible")
    plt.plot(combined_results['Date'], combined_results[f"I_{age_group}"], label="Infectious")
    plt.plot(combined_results['Date'], combined_results[f"R_{age_group}"], label="Recovered")
    plt.plot(combined_results['Date'], combined_results[f"V_{age_group}"], label="Vaccinated")
    
    plt.title(f"SIRV Model for Age Group {age_group}")
    plt.xlabel("Date")
    plt.ylabel("Population")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print("Simulation and plotting complete!")








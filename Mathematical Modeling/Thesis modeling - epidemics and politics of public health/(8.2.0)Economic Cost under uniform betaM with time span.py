# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 06:24:57 2025

@author: joonc
Economic Cost Analysis for an Epidemiological Simulation with Log-Scale Bar Graphs.

This code simulates the epidemic under different social distancing measures (modeled
by scaling the transmission matrix with various β multipliers) and computes the 
associated economic costs. Economic costs include:
  - Medical cost (assuming all infections are mild)
  - Wage loss (adults and seniors, including caregiving losses for children)
  - Death cost (using age-specific mortality fractions and funeral costs)
  - GDP loss (modeled as an exponential function of the β multiplier, shown as positive cost)

Results are stored in a pandas DataFrame, printed to the console, and visualized as grouped bar charts
with y-axes in logarithmic scale.

Key point: We set `time_span = 180` and use `days_intervention = time_span` in the GDP loss calculation.
"""

import os
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Create directory to save figures if it doesn't exist
if not os.path.exists("Figures"):
    os.makedirs("Figures")

# -----------------------------
# 1. Load Parameters and Data
# -----------------------------
def load_parameters(file_path):
    """
    Loads parameters from an Excel file. If your file has a different format,
    adjust the indexing accordingly.
    """
    df = pd.read_excel(file_path, header=None)
    recovery_rates = df.iloc[4, 0:3].values     # gamma for each age group
    waning_immunity_rate = df.iloc[8, 0]          # W
    # We'll ignore the time span from the file; we'll set our own below.
    population_size = df.iloc[14, 0:3].values
    susceptible_init = df.iloc[14, 0:3].values
    infectious_init = df.iloc[16, 0:3].values  
    recovered_init = df.iloc[18, 0:3].values   
    vaccinated_init = df.iloc[20, 0:3].values  
    return (recovery_rates, waning_immunity_rate, 
            population_size, susceptible_init, 
            infectious_init, recovered_init, vaccinated_init)

# Example: Adjust file path as needed
file_path = 'Inputs.xlsx'
(gamma, W, N, S_init, I_init, R_init, V_init) = load_parameters(file_path)

# We define the simulation to run for 180 days, overriding anything in the Excel.
time_span = 1000

# Load transmission rates matrix from CSV (adjust path if needed)
beta_df = pd.read_csv('DataSets/Fitted_Beta_Matrix.csv')
beta = beta_df.iloc[0:3, 1:4].values

# Define initial conditions for 3 groups (Children, Adults, Seniors)
initial_conditions = [
    S_init[0], I_init[0], R_init[0], V_init[0],
    S_init[1], I_init[1], R_init[1], V_init[1],
    S_init[2], I_init[2], R_init[2], V_init[2]
]

# -----------------------------
# 2. Define the SIRV Model (No Vaccination)
# -----------------------------
def deriv(y, t, N, beta, gamma, W):
    """
    SIRV model ODEs for three groups (Children, Adults, Seniors).
    Here, vaccination is set to zero (V remains constant) to focus on economic analysis.
    """
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y
    
    # Force of infection for each group
    lambda1 = beta[0, 0] * I1 / N[0] + beta[0, 1] * I2 / N[1] + beta[0, 2] * I3 / N[2]
    lambda2 = beta[1, 0] * I1 / N[0] + beta[1, 1] * I2 / N[1] + beta[1, 2] * I3 / N[2]
    lambda3 = beta[2, 0] * I1 / N[0] + beta[2, 1] * I2 / N[1] + beta[2, 2] * I3 / N[2]
    
    # Children (group 1)
    dS1dt = -lambda1 * S1 + W * R1
    dI1dt = lambda1 * S1 - gamma[0] * I1
    dR1dt = gamma[0] * I1 - W * R1
    dV1dt = 0  # No vaccination in this scenario
    
    # Adults (group 2)
    dS2dt = -lambda2 * S2 + W * R2
    dI2dt = lambda2 * S2 - gamma[1] * I2
    dR2dt = gamma[1] * I2 - W * R2
    dV2dt = 0
    
    # Seniors (group 3)
    dS3dt = -lambda3 * S3 + W * R3
    dI3dt = lambda3 * S3 - gamma[2] * I3
    dR3dt = gamma[2] * I3 - W * R3
    dV3dt = 0
    
    return [dS1dt, dI1dt, dR1dt, dV1dt,
            dS2dt, dI2dt, dR2dt, dV2dt,
            dS3dt, dI3dt, dR3dt, dV3dt]

def run_simulation(beta_mod):
    """
    Runs the simulation for the given modified beta matrix.
    Returns the simulation results and the time vector.
    """
    t = np.linspace(0, time_span, int(time_span))
    results = odeint(deriv, initial_conditions, t, args=(N, beta_mod, gamma, W))
    return results, t

# -----------------------------
# 3. Economic Cost Analysis Functions
# -----------------------------
# Economic parameters (from references like Kim et al., 2022, or user-provided)
M_mild = 160.8
employment_rates = np.array([0.02, 0.694, 0.513])  # [Children, Adults, Seniors]
daily_income = np.array([27.74, 105.12, 92.45])    # [Children, Adults, Seniors]
funeral_cost = 9405.38

# Proposed mortality rates (converted from percentages):
#   Children: 0.0001% → 0.000001, Adults: 0.003% → 0.00003, Seniors: 0.7% → 0.007
mortality_fraction = np.array([0.000001, 0.00003, 0.007])

# If vaccination is off, cost is effectively zero. We keep the variable for clarity.
vaccination_cost_per_capita = 35.76  

# GDP parameters
GDP_per_capita = 31929
a, b, c = -0.3648, -8.7989, -0.0012  # Exponential model parameters

def calculate_total_cost(results, beta_m, days_intervention, N):
    """
    Computes economic costs based on the simulation output.
    GDP loss is adjusted to be a positive cost, scaled by (days_intervention / 365).
    """
    # Extract compartments
    S = results[:, [0, 4, 8]]
    I = results[:, [1, 5, 9]]
    R = results[:, [2, 6, 10]]
    V = results[:, [3, 7, 11]]
    
    # Medical cost: assume all infections are mild
    medical_cost = M_mild * np.sum(I)
    
    # Wage loss (adults + seniors) + caregiving for children
    wage_loss_adults_seniors = np.sum((daily_income[1:] * employment_rates[1:]) * np.sum(I[:,1:], axis=0))
    adult_avg_income = daily_income[1]
    adult_employment_rate = employment_rates[1]
    wage_loss_caregiving = adult_avg_income * adult_employment_rate * np.sum(I[:,0])
    total_wage_loss = wage_loss_adults_seniors + wage_loss_caregiving
    
    # Death cost (funeral costs for final recovered ~ final infected)
    deaths = mortality_fraction * R[-1, :]
    death_cost = funeral_cost * np.sum(deaths)
    
    # Vaccination cost (set to zero effectively if no vaccination is happening)
    vaccination_cost = vaccination_cost_per_capita * np.sum(V[-1, :])
    
    # GDP loss
    GDP_loss_fraction = a * np.exp(b * beta_m) + c
    GDP_loss = GDP_per_capita * GDP_loss_fraction * (days_intervention / 365) * np.sum(N)
    GDP_loss = abs(GDP_loss)  # Ensure positive
    
    total_cost = medical_cost + total_wage_loss + death_cost + vaccination_cost + GDP_loss
    
    return {
        "Medical Cost": medical_cost,
        "Wage Loss": total_wage_loss,
        "Death Cost": death_cost,
        "Vaccination Cost": vaccination_cost,
        "GDP Loss": GDP_loss,
        "Total Cost": total_cost
    }

# -----------------------------
# 4. Simulation Loop for Social Distancing Scenarios
# -----------------------------
beta_multipliers = [1.0, 0.7, 0.35, 0.2]
results_list = []

for multiplier in beta_multipliers:
    # Modify beta
    beta_mod = beta * multiplier
    
    # Run simulation
    sim_results, t = run_simulation(beta_mod)
    
    # Calculate costs using time_span as the intervention length
    costs = calculate_total_cost(
        sim_results, 
        beta_m=multiplier, 
        days_intervention=time_span,  # Use time_span (180 days)
        N=N
    )
    
    # For reporting: peak infections & day of peak
    I_total = sim_results[:, 1] + sim_results[:, 5] + sim_results[:, 9]
    peak_value = np.max(I_total)
    peak_day = t[np.argmax(I_total)]
    
    result = {
        "beta_multiplier": multiplier,
        "peak_infected": peak_value,
        "peak_day": peak_day
    }
    result.update(costs)
    results_list.append(result)

# Convert results to DataFrame
results_df = pd.DataFrame(results_list)
print("Social Distancing Scenario with Economic Costs (time_span = 180 days):")
print(results_df)

# -----------------------------
# 5. Visualizations (Log Scale Bar Charts)
# -----------------------------
# Grouped Bar Chart for Cost Components
cost_components = ["Medical Cost", "Wage Loss", "Death Cost", "GDP Loss"]
n_groups = len(results_df)
index = np.arange(n_groups)
bar_width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))

for i, component in enumerate(cost_components):
    ax.bar(index + i * bar_width, results_df[component], bar_width, label=component)

ax.set_xlabel('Beta Multiplier')
ax.set_ylabel('Cost in USD (Log Scale)')
ax.set_title(f'Economic Cost Breakdown by Beta Multiplier (Time Span: {time_span} days, Log Scale)')
ax.set_xticks(index + bar_width * (len(cost_components) - 1) / 2)
ax.set_xticklabels(results_df['beta_multiplier'])
ax.legend()
ax.set_yscale('log')
ax.grid(True, which='both', ls='--')

# Save the grouped bar chart with time_span info in the filename
plt.savefig(f"Figures/Economic_Cost_Breakdown_{time_span}days.png")
plt.show()

# Line Graph for Total Economic Cost vs. Beta Multiplier
plt.figure(figsize=(10, 6))
plt.plot(results_df['beta_multiplier'], results_df['Total Cost'], marker='o', linestyle='-', color='blue')
plt.xlabel('Beta Multiplier')
plt.ylabel('Total Economic Cost in USD (Log Scale)')
plt.title(f'Total Economic Cost vs Social Distancing (Beta Multiplier) (Time Span: {time_span} days, Log Scale)')
plt.yscale('log')
plt.grid(True, which='both', ls='--')

# Save the line graph with time_span info in the filename
plt.savefig(f"Figures/Total_Economic_Cost_{time_span}days.png")
plt.show()

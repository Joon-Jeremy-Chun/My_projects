# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 23:23:34 2024

@author: joonc
"""

import os
import pandas as pd
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Create directory for figures if it doesn't exist
fig_dir = 'Figures'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

# Step 1: Load Data and Fixed Parameters
file_path_parameters = 'DataSets/Korea_threeGroups_SIRV_parameter.csv'
file_path_inputs = 'Inputs.xlsx'

# Load real-world data
real_data = pd.read_csv(file_path_parameters)
real_data['Date'] = pd.to_datetime(real_data['Date'])

# Load fixed parameters
inputs = pd.read_excel(file_path_inputs, header=None)
initial_beta = inputs.iloc[0:3, 0:3].values 
recovery_rates = inputs.iloc[4, 0:3].values
vaccination_rates = inputs.iloc[10, 0:3].values
waning_immunity_rate = inputs.iloc[8, 0]

# Age groups
age_groups = ['0-19', '20-49', '50-80+']
gamma = recovery_rates
a = vaccination_rates
delta = waning_immunity_rate

# Step 2: Select Start and End Dates for Fitting
start_date = '2020-02-01'  # Start date
end_date = '2020-03-01'    # End date

# Filter real-world data for the specified date range
fit_data = real_data[(real_data['Date'] >= start_date) & (real_data['Date'] <= end_date)].reset_index(drop=True)

# Check if data is empty
if fit_data.empty:
    raise ValueError("No data found between the specified start_date and end_date. Please check the date range.")

# Step 3: Define SIRV Model ODEs
def sirv_model(y, t, beta, gamma, a, delta):
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y

    # Force of infection (lambda)
    N = [S1 + I1 + R1 + V1, S2 + I2 + R2 + V2, S3 + I3 + R3 + V3]
    lambda1 = sum(beta[0, j] * Ij / Nj for j, Ij, Nj in zip(range(3), [I1, I2, I3], N))
    lambda2 = sum(beta[1, j] * Ij / Nj for j, Ij, Nj in zip(range(3), [I1, I2, I3], N))
    lambda3 = sum(beta[2, j] * Ij / Nj for j, Ij, Nj in zip(range(3), [I1, I2, I3], N))

    # Differential equations
    dS1 = -lambda1 * S1 - a[0] * S1 + delta * (R1 + V1)
    dI1 = lambda1 * S1 - gamma[0] * I1
    dR1 = gamma[0] * I1 - delta * R1
    dV1 = a[0] * S1 - delta * V1

    dS2 = -lambda2 * S2 - a[1] * S2 + delta * (R2 + V2)
    dI2 = lambda2 * S2 - gamma[1] * I2
    dR2 = gamma[1] * I2 - delta * R2
    dV2 = a[1] * S2 - delta * V2

    dS3 = -lambda3 * S3 - a[2] * S3 + delta * (R3 + V3)
    dI3 = lambda3 * S3 - gamma[2] * I3
    dR3 = gamma[2] * I3 - delta * R3
    dV3 = a[2] * S3 - delta * V3

    return [dS1, dI1, dR1, dV1, dS2, dI2, dR2, dV2, dS3, dI3, dR3, dV3]

# Step 4: Objective Function to Minimize
def objective(beta_flat):
    beta = beta_flat.reshape((3, 3))
    y0 = [
        fit_data['S_0-19'][0], fit_data['I_0-19'][0], fit_data['R_0-19'][0], fit_data['V_0-19'][0],
        fit_data['S_20-49'][0], fit_data['I_20-49'][0], fit_data['R_20-49'][0], fit_data['V_20-49'][0],
        fit_data['S_50-80+'][0], fit_data['I_50-80+'][0], fit_data['R_50-80+'][0], fit_data['V_50-80+'][0]
    ]
    t = range(len(fit_data))

    # Solve ODEs
    result = odeint(sirv_model, y0, t, args=(beta, gamma, a, delta))
    I_model = result[:, [1, 5, 9]]  # Extract Infectious compartments

    # Compare model output to observed data
    I_real = fit_data[['I_0-19', 'I_20-49', 'I_50-80+']].values
    error = np.mean((I_model - I_real) ** 2)
    return error

# Step 5: Optimize Transmission Matrix Beta
initial_beta = inputs.iloc[0:3, 0:3].values  # Initial guess for beta
bounds = [(0, None)] * 9  # Bounds: 0 to infinity for all 9 beta values (flattened 3x3 matrix)

result = minimize(objective, initial_beta.flatten(), method='L-BFGS-B', bounds=bounds)

# Check if optimization was successful
if result.success:
    optimal_beta = result.x.reshape((3, 3))
    print("Optimization Successful!")
else:
    print("Optimization Failed:", result.message)
    exit()

# Step 6: Solve the Model with Fitted Beta
y0 = [
    fit_data['S_0-19'][0], fit_data['I_0-19'][0], fit_data['R_0-19'][0], fit_data['V_0-19'][0],
    fit_data['S_20-49'][0], fit_data['I_20-49'][0], fit_data['R_20-49'][0], fit_data['V_20-49'][0],
    fit_data['S_50-80+'][0], fit_data['I_50-80+'][0], fit_data['R_50-80+'][0], fit_data['V_50-80+'][0]
]
t = range(len(fit_data))
fitted_result = odeint(sirv_model, y0, t, args=(optimal_beta, gamma, a, delta))
I_fitted = fitted_result[:, [1, 5, 9]]

# Step 7: Plot Results for Each Age Group and Save Figures
for idx, age_group in enumerate(age_groups):
    plt.figure(figsize=(10, 6))
    plt.plot(fit_data['Date'], fit_data[f'I_{age_group}'], label='Actual I(t)', color='blue', linestyle='--')
    plt.plot(fit_data['Date'], I_fitted[:, idx], label='Fitted I(t)', color='red')
    plt.title(f"Actual vs Fitted I(t) for Age Group {age_group}")
    plt.xlabel("Date")
    plt.ylabel("Infectious Population")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    # Save the figure to the Figures directory
    save_path = os.path.join(fig_dir, f"Actual_vs_Fitted_I_{age_group.replace('-', '_')}.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

# Step 7A: Plot Total I(t) Across All Age Groups and Save Figure
# Calculate total I(t) from the actual data:
total_actual_I = (
    fit_data['I_0-19'] +
    fit_data['I_20-49'] +
    fit_data['I_50-80+']
)

# Calculate total I(t) from the fitted model:
total_fitted_I = I_fitted[:, 0] + I_fitted[:, 1] + I_fitted[:, 2]

plt.figure(figsize=(10, 6))
plt.plot(fit_data['Date'], total_actual_I, label='Actual Total I(t)', color='blue', linestyle='--')
plt.plot(fit_data['Date'], total_fitted_I, label='Fitted Total I(t)', color='red')
plt.title("Actual vs Fitted Total Infectious (All Age Groups)")
plt.xlabel("Date")
plt.ylabel("Infectious Population")
plt.legend()
plt.grid()
plt.tight_layout()

# Save the total I(t) plot
save_path_total = os.path.join(fig_dir, "Actual_vs_Fitted_Total_I.png")
plt.savefig(save_path_total, dpi=300)
plt.show()

# Step 8: Save Results
output_file = 'DataSets/Fitted_Beta_Matrix.csv'
print(optimal_beta)
pd.DataFrame(optimal_beta, columns=age_groups, index=age_groups).to_csv(output_file, index=True)
print(f"Fitted beta matrix saved to: {output_file}")

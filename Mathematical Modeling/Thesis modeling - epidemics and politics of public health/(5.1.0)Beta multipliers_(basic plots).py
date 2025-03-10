# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 01:40:39 2025

@author: joonc
"""
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# -------------------------------
# 1. Define the SIRV model ODEs
# -------------------------------
def sirv_model(y, t, beta, gamma, a, delta):
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y
    # Compute total population per group
    N1 = S1 + I1 + R1 + V1
    N2 = S2 + I2 + R2 + V2
    N3 = S3 + I3 + R3 + V3
    # Force of infection for each group
    lambda1 = sum(beta[0, j] * I / N for j, I, N in zip(range(3), [I1, I2, I3], [N1, N2, N3]))
    lambda2 = sum(beta[1, j] * I / N for j, I, N in zip(range(3), [I1, I2, I3], [N1, N2, N3]))
    lambda3 = sum(beta[2, j] * I / N for j, I, N in zip(range(3), [I1, I2, I3], [N1, N2, N3]))
    
    # Differential equations for group 1
    dS1 = -lambda1 * S1 - a[0] * S1 + delta * (R1 + V1)
    dI1 = lambda1 * S1 - gamma[0] * I1
    dR1 = gamma[0] * I1 - delta * R1
    dV1 = a[0] * S1 - delta * V1
    
    # Differential equations for group 2
    dS2 = -lambda2 * S2 - a[1] * S2 + delta * (R2 + V2)
    dI2 = lambda2 * S2 - gamma[1] * I2
    dR2 = gamma[1] * I2 - delta * R2
    dV2 = a[1] * S2 - delta * V2
    
    # Differential equations for group 3
    dS3 = -lambda3 * S3 - a[2] * S3 + delta * (R3 + V3)
    dI3 = lambda3 * S3 - gamma[2] * I3
    dR3 = gamma[2] * I3 - delta * R3
    dV3 = a[2] * S3 - delta * V3
    
    return [dS1, dI1, dR1, dV1,
            dS2, dI2, dR2, dV2,
            dS3, dI3, dR3, dV3]

# -------------------------------
# 2. Load Data and Parameters
# -------------------------------

# (a) Load the full real-world data from Korea CSV (includes dates beyond fitting period if available)
data_file = 'DataSets/Korea_threeGroups_SIRV_parameter.csv'
real_data = pd.read_csv(data_file)
real_data['Date'] = pd.to_datetime(real_data['Date'])

# (b) Define the fitting period (used to set initial conditions)
fit_start_date = '2020-02-01'
fit_end_date = '2020-03-01'
fit_data = real_data[(real_data['Date'] >= fit_start_date) & (real_data['Date'] <= fit_end_date)].reset_index(drop=True)
if fit_data.empty:
    raise ValueError("No data found between the specified dates for fitting.")

# (c) Load fixed parameters from Inputs.xlsx
inputs = pd.read_excel('Inputs.xlsx', header=None)
recovery_rates = inputs.iloc[4, 0:3].values      # gamma (recovery rates)
vaccination_rates = inputs.iloc[10, 0:3].values    # a (vaccination rates)
waning_immunity_rate = inputs.iloc[8, 0]           # delta (waning immunity rate)
gamma = recovery_rates
a = vaccination_rates
delta = waning_immunity_rate

# (d) Load the fitted beta matrix from CSV (optimal beta saved from parameter fitting)
optimal_beta_df = pd.read_csv('DataSets/Fitted_Beta_Matrix.csv', index_col=0)
optimal_beta = optimal_beta_df.values  # (3 x 3) matrix

# Define age groups for labeling (ensure names match your CSV columns)
age_groups = ['0-19', '20-49', '50-80+']

# (e) Set initial conditions from the first row of the fitting period
y0 = [
    fit_data['S_0-19'][0], fit_data['I_0-19'][0], fit_data['R_0-19'][0], fit_data['V_0-19'][0],
    fit_data['S_20-49'][0], fit_data['I_20-49'][0], fit_data['R_20-49'][0], fit_data['V_20-49'][0],
    fit_data['S_50-80+'][0], fit_data['I_50-80+'][0], fit_data['R_50-80+'][0], fit_data['V_50-80+'][0]
]

# -------------------------------
# 3. Extended Simulation
# -------------------------------
fitting_days = len(fit_data)    # Number of days in the fitting period
extra_days = 10                 # Extend simulation by 90 days (~3 months)
total_days = fitting_days + extra_days
t_extended = np.arange(total_days)  # time grid (in days)

# Solve the SIRV model with the fitted beta over the extended period
fitted_result_extended = odeint(sirv_model, y0, t_extended, args=(optimal_beta, gamma, a, delta))

# Extract infectious compartments for each age group: indices 1, 5, and 9.
I_fitted_extended = np.column_stack((
    fitted_result_extended[:, 1],
    fitted_result_extended[:, 5],
    fitted_result_extended[:, 9]
))
# Total infectious from fitted simulation (sum over groups)
I_fitted_total = np.sum(I_fitted_extended, axis=1)

# Generate simulation dates starting from the first date of the fitting period
start_date_dt = fit_data['Date'].iloc[0]
simulation_dates = [start_date_dt + pd.Timedelta(days=int(i)) for i in t_extended]

# -------------------------------
# 4. Prepare Actual Data for Extended Plotting
# -------------------------------
# Here we use all available actual data (even beyond the fitting period) if available.
# Filter the real_data to only include dates within our simulation period.
actual_data_extended = real_data[(real_data['Date'] >= simulation_dates[0]) & 
                                 (real_data['Date'] <= simulation_dates[-1])].copy()
# Extract actual I(t) for each age group (ensure column names match)
I_actual_extended = np.column_stack((
    actual_data_extended['I_0-19'].values,
    actual_data_extended['I_20-49'].values,
    actual_data_extended['I_50-80+'].values
))
# Total actual infectious (if available)
I_actual_total_extended = np.sum(I_actual_extended, axis=1)

# -------------------------------
# 5. Plotting: Age Groups and Total (Actual vs. Fitted)
# -------------------------------

import matplotlib.dates as mdates

# For each age group: plot fitted simulation over extended period and overlay actual data.
for idx, group in enumerate(age_groups):
    plt.figure(figsize=(10, 6))
    # Plot fitted simulation (red line)
    plt.plot(simulation_dates, I_fitted_extended[:, idx], 'r-', label='Fitted I(t)')
    # Plot actual data (blue markers) if available
    plt.plot(actual_data_extended['Date'], actual_data_extended[f'I_{group}'], 'b--', label='Actual I(t)')
    # Mark the last date of actual available data
    if not actual_data_extended.empty:
        last_actual_date = actual_data_extended['Date'].max()
        plt.axvline(x=last_actual_date, color='k', linestyle=':', label='Last Actual Data')
    plt.title(f'Actual vs Fitted I(t) for Age Group {group}')
    plt.xlabel('Date')
    plt.ylabel('Infectious Population')
    plt.legend()
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.show()

# Plot for Total Infectious Population (sum over groups)
plt.figure(figsize=(10, 6))
plt.plot(simulation_dates, I_fitted_total, 'r-', label='Fitted Total I(t)')
plt.plot(actual_data_extended['Date'], I_actual_total_extended, 'b--', label='Actual Total I(t)')
if not actual_data_extended.empty:
    last_actual_date = actual_data_extended['Date'].max()
    plt.axvline(x=last_actual_date, color='k', linestyle=':', label='Last Actual Data')
plt.title('Actual vs Fitted Total Infectious Population')
plt.xlabel('Date')
plt.ylabel('Total Infectious Population')
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.show()


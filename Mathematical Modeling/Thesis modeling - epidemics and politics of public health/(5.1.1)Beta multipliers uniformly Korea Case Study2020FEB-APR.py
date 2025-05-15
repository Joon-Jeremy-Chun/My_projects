# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 02:10:33 2025

@author: joonc

Script to simulate intervention scenarios using β reduction multipliers and to produce four plots:
1. Infectious population for 0-19,
2. Infectious population for 20-49,
3. Infectious population for 50-80+,
4. Total infectious population.
The simulation uses an intervention on February 27, after which the fitted β matrix is multiplied uniformly by candidate multipliers.

Intervention Simulation with Candidate Beta Multipliers and Restricted Real-Data Plots

This script:
1. Runs the SIRV model simulation splitting pre- and post-intervention (Feb 27).
2. Uses candidate multipliers [0.4, 0.3, 0.2] to adjust the beta matrix post-intervention.
3. Restricts real data to the simulation domain for plotting and peak computation.
4. Plots side-by-side comparisons for each age group (0-19, 20-49, 50-80+) and total.
5. Computes peaks (maximum infectious counts) for both real data and simulations, within the same date range.
6. Saves peak information in a CSV.

Adjust file paths, date ranges, etc., as needed.
"""

import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# -------------------------------
# 1. Define the SIRV Model ODEs
# -------------------------------
def sirv_model(y, t, beta, gamma, a, delta):
    """
    SIRV model with three groups:
      y = [S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3]
    beta: 3x3 transmission rate matrix
    gamma: array of length 3 (recovery rates)
    a: array of length 3 (vaccination rates)
    delta: scalar (waning immunity rate)
    """
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y
    
    # Population sizes in each group
    N1 = S1 + I1 + R1 + V1
    N2 = S2 + I2 + R2 + V2
    N3 = S3 + I3 + R3 + V3
    
    # Force of infection for each group
    lambda1 = sum(beta[0, j] * I / N for j, I, N in zip(range(3), [I1, I2, I3], [N1, N2, N3]))
    lambda2 = sum(beta[1, j] * I / N for j, I, N in zip(range(3), [I1, I2, I3], [N1, N2, N3]))
    lambda3 = sum(beta[2, j] * I / N for j, I, N in zip(range(3), [I1, I2, I3], [N1, N2, N3]))
    
    # Group 1 (0-19)
    dS1 = -lambda1 * S1 - a[0] * S1 + delta * (R1 + V1)
    dI1 = lambda1 * S1 - gamma[0] * I1
    dR1 = gamma[0] * I1 - delta * R1
    dV1 = a[0] * S1 - delta * V1
    
    # Group 2 (20-49)
    dS2 = -lambda2 * S2 - a[1] * S2 + delta * (R2 + V2)
    dI2 = lambda2 * S2 - gamma[1] * I2
    dR2 = gamma[1] * I2 - delta * R2
    dV2 = a[1] * S2 - delta * V2
    
    # Group 3 (50-80+)
    dS3 = -lambda3 * S3 - a[2] * S3 + delta * (R3 + V3)
    dI3 = lambda3 * S3 - gamma[2] * I3
    dR3 = gamma[2] * I3 - delta * R3
    dV3 = a[2] * S3 - delta * V3
    
    return [
        dS1, dI1, dR1, dV1,
        dS2, dI2, dR2, dV2,
        dS3, dI3, dR3, dV3
    ]

# -------------------------------
# 2. Load Data and Parameters
# -------------------------------
# (a) Load full real-world data
data_file = 'DataSets/Korea_threeGroups_SIRV_parameter.csv'
real_data = pd.read_csv(data_file)
real_data['Date'] = pd.to_datetime(real_data['Date'])

# (b) Define the fitting period (for initial conditions)
fit_start_date = '2020-02-01'
fit_end_date   = '2020-03-01'
fit_data = real_data[(real_data['Date'] >= fit_start_date) & (real_data['Date'] <= fit_end_date)].reset_index(drop=True)
if fit_data.empty:
    raise ValueError("No data found between the specified dates for fitting.")

# (c) Load fixed parameters from Inputs.xlsx
inputs = pd.read_excel('Inputs.xlsx', header=None)
recovery_rates    = inputs.iloc[4, 0:3].values       # gamma
vaccination_rates = inputs.iloc[10, 0:3].values      # a
waning_immunity_rate = inputs.iloc[8, 0]             # delta

gamma = recovery_rates
a = vaccination_rates
delta = waning_immunity_rate

# (d) Load the fitted beta matrix (3x3) from CSV
optimal_beta_df = pd.read_csv('DataSets/Fitted_Beta_Matrix.csv', index_col=0)
optimal_beta = optimal_beta_df.values

# Define age groups for labeling
age_groups = ['0-19', '20-49', '50-80+']

# (e) Set initial conditions from the first day of the fitting period
y0 = [
    fit_data['S_0-19'][0], fit_data['I_0-19'][0], fit_data['R_0-19'][0], fit_data['V_0-19'][0],
    fit_data['S_20-49'][0], fit_data['I_20-49'][0], fit_data['R_20-49'][0], fit_data['V_20-49'][0],
    fit_data['S_50-80+'][0], fit_data['I_50-80+'][0], fit_data['R_50-80+'][0], fit_data['V_50-80+'][0]
]

# -------------------------------
# 3. Define Simulation Period
# -------------------------------
fitting_days = len(fit_data)  # days in the fitting period
extra_days = 120            # extend by 90 days
total_days = fitting_days + extra_days
t_extended = np.arange(total_days)  # 0,1,2,..., total_days-1

# Generate simulation dates starting from the first fitting date
start_date_dt = fit_data['Date'].iloc[0]
simulation_dates = [start_date_dt + pd.Timedelta(days=int(i)) for i in t_extended]

# -------------------------------
# 4. Run Intervention Simulation
# -------------------------------
# Government intervention intensifies on Feb 27
intervention_date = pd.Timestamp('2020-02-27')
# Find index for the intervention date
intervention_idx = next(i for i, d in enumerate(simulation_dates) if d >= intervention_date)

# Pre-intervention simulation (day 0 through intervention day)
t_pre = t_extended[:intervention_idx+1]
sim_pre = odeint(sirv_model, y0, t_pre, args=(optimal_beta, gamma, a, delta))

# Candidate beta multipliers
candidate_multipliers = [0.27, 0.2]

results_intervention = {}
for mult in candidate_multipliers:
    # Scale beta uniformly
    beta_new = mult * optimal_beta
    # Last state of pre-intervention = initial condition for post-intervention
    y_intervention_init = sim_pre[-1]
    t_post = t_extended[intervention_idx:]
    sim_post = odeint(sirv_model, y_intervention_init, t_post, args=(beta_new, gamma, a, delta))
    # Combine (avoid duplicating the intervention day)
    sim_full = np.vstack((sim_pre, sim_post[1:]))
    results_intervention[mult] = sim_full

# Extract Infectious compartments
sim_I = {}
total_infectious = {}
for mult, sim in results_intervention.items():
    group_I = {
        '0-19': sim[:, 1],
        '20-49': sim[:, 5],
        '50-80+': sim[:, 9]
    }
    sim_I[mult] = group_I
    total_infectious[mult] = sim[:, 1] + sim[:, 5] + sim[:, 9]

# -------------------------------
# 5. Restrict Real Data to Simulation Domain for Plotting
# -------------------------------
plot_data = real_data[(real_data['Date'] >= simulation_dates[0]) & 
                      (real_data['Date'] <= simulation_dates[-1])].copy()

# -------------------------------
# 6. Plot Simulation vs. Actual Data (Restricted Domain)
# -------------------------------
import matplotlib.dates as mdates

# Plot for each age group
for group in age_groups:
    plt.figure(figsize=(10, 6))
    # Simulation curves
    for mult in candidate_multipliers:
        plt.plot(simulation_dates, sim_I[mult][group], label=f'β Multiplier = {mult}')
    # Actual data (restricted to simulation domain)
    plt.plot(plot_data['Date'], plot_data[f'I_{group}'], 'k--', label=f'Actual I(t) for {group}')
    plt.axvline(x=intervention_date, color='gray', linestyle=':', label='Intervention (Feb 27)')
    plt.title(f'Infectious Population for Age Group {group}')
    plt.xlabel('Date')
    plt.ylabel('Infectious Population')
    plt.legend()
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    # Optionally limit the x-axis to show only the simulation range
    plt.xlim([simulation_dates[0], simulation_dates[-1]])
    plt.show()

# Plot for the Total Infectious Population (sum over groups)
plt.figure(figsize=(10, 6))
for mult in candidate_multipliers:
    plt.plot(simulation_dates, total_infectious[mult], label=f'β Multiplier = {mult}')
# Actual total (restricted domain)
I_actual_total = (
    plot_data['I_0-19'].values +
    plot_data['I_20-49'].values +
    plot_data['I_50-80+'].values
)
plt.plot(plot_data['Date'], I_actual_total, 'k--', label='Actual Total I(t)')
plt.axvline(x=intervention_date, color='gray', linestyle=':', label='Intervention (Feb 27)')
plt.title('Total Infectious Population (All Age Groups)')
plt.xlabel('Date')
plt.ylabel('Total Infectious Population')
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.xlim([simulation_dates[0], simulation_dates[-1]])
plt.show()

# -------------------------------
# 7. Compute Peaks within Simulation Domain & Save to CSV
# -------------------------------
results_list = []

# Real data peaks in the simulation domain
for group in age_groups:
    series = plot_data[f'I_{group}'].values
    dates = plot_data['Date'].values
    peak_idx = np.argmax(series)
    peak_val = series[peak_idx]
    peak_date = dates[peak_idx]
    results_list.append({
        'Data Source': 'Real',
        'Age Group': group,
        'Peak Infectious': peak_val,
        'Peak Date': peak_date
    })

# Total for real data
series_total = (
    plot_data['I_0-19'].values +
    plot_data['I_20-49'].values +
    plot_data['I_50-80+'].values
)
peak_idx_total = np.argmax(series_total)
peak_val_total = series_total[peak_idx_total]
peak_date_total = plot_data['Date'].values[peak_idx_total]
results_list.append({
    'Data Source': 'Real',
    'Age Group': 'Total',
    'Peak Infectious': peak_val_total,
    'Peak Date': peak_date_total
})

# Simulation peaks
for mult in candidate_multipliers:
    sim_data = np.array(results_intervention[mult])
    for group, idx in zip(age_groups, [1, 5, 9]):
        series = sim_data[:, idx]
        peak_idx = np.argmax(series)
        peak_val = series[peak_idx]
        peak_date = simulation_dates[peak_idx]
        results_list.append({
            'Data Source': f'β Multiplier = {mult}',
            'Age Group': group,
            'Peak Infectious': peak_val,
            'Peak Date': peak_date
        })
    # Total
    series_total_sim = sim_data[:, 1] + sim_data[:, 5] + sim_data[:, 9]
    peak_idx_total_sim = np.argmax(series_total_sim)
    peak_val_total_sim = series_total_sim[peak_idx_total_sim]
    peak_date_total_sim = simulation_dates[peak_idx_total_sim]
    results_list.append({
        'Data Source': f'β Multiplier = {mult}',
        'Age Group': 'Total',
        'Peak Infectious': peak_val_total_sim,
        'Peak Date': peak_date_total_sim
    })

peaks_df = pd.DataFrame(results_list)
print("Peak Infectious Values and Dates (restricted to simulation domain):")
print(peaks_df)

# Save peaks info
peaks_df.to_csv('DataSets/Peak_Infectious_Comparisons.csv', index=False)

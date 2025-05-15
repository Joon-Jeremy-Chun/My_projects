# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 03:22:53 2025

@author: joonc

Rolling Optimization of 3x3 Beta Multiplier Matrix (Daily)
-----------------------------------------------------------
For each day in a 30-day period, using the state from real data as the initial condition,
we simulate one day ahead using a modified beta matrix:
    new_beta = M ⊙ fitted_beta   (element-wise multiplication)
We then optimize M (a 3x3 matrix, 9 parameters) so that the simulation result at t=1 
matches the next day’s observed state as closely as possible (minimizing SSE).
The optimal multiplier for each day is recorded, and finally we plot the time evolution
of each element of the 3x3 multiplier matrix.
"""

import pandas as pd
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# -------------------------------
# 1. Define the SIRV Model ODEs
# -------------------------------
def sirv_model(y, t, beta, gamma, a, delta):
    """
    SIRV model with three age groups.
    y = [S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3]
    beta: 3x3 transmission rate matrix (applied in force-of-infection)
    gamma: recovery rates (array, length 3)
    a: vaccination rates (array, length 3)
    delta: waning immunity rate (scalar)
    """
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y

    # Compute population sizes
    N1 = S1 + I1 + R1 + V1
    N2 = S2 + I2 + R2 + V2
    N3 = S3 + I3 + R3 + V3

    # Force of infection for each group:
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

    return [dS1, dI1, dR1, dV1,
            dS2, dI2, dR2, dV2,
            dS3, dI3, dR3, dV3]

# -------------------------------
# 2. Load Real Data and Fixed Parameters
# -------------------------------
# Load real data; adjust file path as needed.
data_file = 'DataSets/Korea_threeGroups_SIRV_parameter.csv'
real_data = pd.read_csv(data_file)
real_data['Date'] = pd.to_datetime(real_data['Date'])

# Choose the period for rolling optimization.
# For example, we need 31 consecutive days (to have 30 optimization steps).
# Let’s assume we use data from '2020-02-26' to '2020-03-28'.
start_roll = pd.Timestamp('2020-03-01')
end_roll = pd.Timestamp('2020-03-28')
roll_data = real_data[(real_data['Date'] >= start_roll) & (real_data['Date'] <= end_roll)].copy()
roll_data = roll_data.sort_values(by='Date').reset_index(drop=True)
n_days = len(roll_data)  # We need n_days - 1 optimization steps.
if n_days < 2:
    raise ValueError("Not enough data in the rolling period.")

# Use the state on Feb 26 (first row) as initial condition for day0.
def get_state_from_row(row):
    # Order: S_0-19, I_0-19, R_0-19, V_0-19, S_20-49, I_20-49, R_20-49, V_20-49, S_50-80+, I_50-80+, R_50-80+, V_50-80+
    return np.array([
        row['S_0-19'], row['I_0-19'], row['R_0-19'], row['V_0-19'],
        row['S_20-49'], row['I_20-49'], row['R_20-49'], row['V_20-49'],
        row['S_50-80+'], row['I_50-80+'], row['R_50-80+'], row['V_50-80+']
    ], dtype=float)

# Fixed parameters from Inputs.xlsx (adjust file path if needed)
inputs = pd.read_excel('Inputs.xlsx', header=None)
recovery_rates    = inputs.iloc[4, 0:3].values  # gamma, length 3
vaccination_rates = inputs.iloc[10, 0:3].values # a, length 3
waning_immunity_rate = inputs.iloc[8, 0]        # delta

gamma = recovery_rates
a = vaccination_rates
delta = waning_immunity_rate

# Load the fitted beta matrix (3x3) from CSV
optimal_beta_df = pd.read_csv('DataSets/Fitted_Beta_Matrix.csv', index_col=0)
optimal_beta = optimal_beta_df.values

# -------------------------------
# 3. Set Rolling Optimization Simulation Settings
# -------------------------------
# For each day, we simulate forward for 1 day (t from 0 to 1)
t_span = [0, 1]

# We'll optimize the multiplier matrix for each day such that:
# starting from the state on day d (from roll_data), simulating one day forward
# using new_beta = M ⊙ optimal_beta gives a result that best fits the state on day d+1.

def objective_day(x, y0_day, y_target, beta_original, gamma, a, delta):
    """
    x: vector of 9 parameters (to be reshaped into a 3x3 multiplier matrix M)
    y0_day: state vector (length 12) for day d (from real data)
    y_target: state vector (length 12) for day d+1 (from real data)
    Returns SSE between simulated state at t=1 and y_target.
    """
    M = x.reshape((3, 3))
    new_beta = M * beta_original  # element-wise multiplication
    sim = odeint(sirv_model, y0_day, t_span, args=(new_beta, gamma, a, delta))
    y_sim = sim[-1]
    error = np.sum((y_sim - y_target)**2)
    return error

# We'll use bounds for each element, e.g., between 0 and 2.
bounds = [(0, 2)] * 9
x0 = np.ones(9)  # initial guess: all multipliers = 1

# -------------------------------
# 4. Rolling Optimization Loop
# -------------------------------
# We'll store the optimal multiplier matrix for each day (for days 0 to n_days-2)
optimal_matrices = []
dates = []

# We'll run for each day d, where target is day d+1.
for d in range(n_days - 1):
    y0_day = get_state_from_row(roll_data.iloc[d])
    y_target = get_state_from_row(roll_data.iloc[d+1])
    # Optimize the 9 parameters:
    res = minimize(objective_day, x0, args=(y0_day, y_target, optimal_beta, gamma, a, delta),
                   method='L-BFGS-B', bounds=bounds)
    if res.success:
        best_M = res.x.reshape((3, 3))
    else:
        best_M = np.full((3, 3), np.nan)
    optimal_matrices.append(best_M)
    dates.append(roll_data['Date'].iloc[d])
    # Optionally update x0 for next day's optimization (warm start)
    x0 = res.x

# Convert the list of 3x3 matrices into a DataFrame with 9 columns.
# We'll flatten each matrix row-wise.
data_rows = []
for d, M in zip(dates, optimal_matrices):
    flat = M.flatten()
    data_rows.append([d] + list(flat))

columns = ['Date', 'M00', 'M01', 'M02',
           'M10', 'M11', 'M12',
           'M20', 'M21', 'M22']

multiplier_df = pd.DataFrame(data_rows, columns=columns)
print("Daily Optimal 3x3 Multiplier Matrices:")
print(multiplier_df)

# Save the DataFrame
multiplier_df.to_csv('DataSets/Daily_Beta_Multipliers.csv', index=False)

# -------------------------------
# 5. Plot 9 Time Series for Each Matrix Element Over the 30 Days
# -------------------------------
for col in columns[1:]:
    plt.figure(figsize=(10, 4))
    plt.plot(multiplier_df['Date'], multiplier_df[col], marker='o', linestyle='-', label=col)
    avg_val = multiplier_df[col].mean()
    plt.axhline(y=avg_val, color='r', linestyle='--', label=f'Mean = {avg_val:.3f}')
    plt.title(f"Time Series of {col} Over 30 Days")
    plt.xlabel("Date")
    plt.ylabel(col)
    plt.legend()
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.show()


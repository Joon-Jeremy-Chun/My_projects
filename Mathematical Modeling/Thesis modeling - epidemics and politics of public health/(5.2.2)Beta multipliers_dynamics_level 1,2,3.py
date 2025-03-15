# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 04:24:25 2025

@author: joonc

Recursive One-Step-Ahead Forecast with Daily 3x3 Beta Multiplier Optimization
Using solve_ivp (BDF method) with relaxed tolerances to mitigate stiffness issues.
Rolling period: March 1 to March 31, 2020.
Produces:
  - 4 plots: one for each age group (0-19, 20-49, 50-80+) and one for the total infectious population.
  - 9 plots: one for each element of the 3x3 multiplier matrix over time.
All figures are saved in the folder 'Figures'.
"""

import os
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Create "Figures" folder if it does not exist
fig_folder = "Figures"
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

# -------------------------------
# 1. Define the SIRV Model ODEs
# -------------------------------
def sirv_model(t, y, beta, gamma, a, delta):
    """
    SIRV model with three age groups.
    y = [S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3]
    beta: 3x3 transmission rate matrix (used element-wise in force-of-infection)
    gamma: recovery rates (array, length 3)
    a: vaccination rates (array, length 3)
    delta: waning immunity rate (scalar)
    """
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y

    N1 = S1 + I1 + R1 + V1
    N2 = S2 + I2 + R2 + V2
    N3 = S3 + I3 + R3 + V3

    lambda1 = sum(beta[0, j] * I / N for j, I, N in zip(range(3), [I1, I2, I3], [N1, N2, N3]))
    lambda2 = sum(beta[1, j] * I / N for j, I, N in zip(range(3), [I1, I2, I3], [N1, N2, N3]))
    lambda3 = sum(beta[2, j] * I / N for j, I, N in zip(range(3), [I1, I2, I3], [N1, N2, N3]))

    dS1 = -lambda1 * S1 - a[0]*S1 + delta*(R1+V1)
    dI1 = lambda1 * S1 - gamma[0]*I1
    dR1 = gamma[0]*I1 - delta*R1
    dV1 = a[0]*S1 - delta*V1

    dS2 = -lambda2 * S2 - a[1]*S2 + delta*(R2+V2)
    dI2 = lambda2 * S2 - gamma[1]*I2
    dR2 = gamma[1]*I2 - delta*R2
    dV2 = a[1]*S2 - delta*V2

    dS3 = -lambda3 * S3 - a[2]*S3 + delta*(R3+V3)
    dI3 = lambda3 * S3 - gamma[2]*I3
    dR3 = gamma[2]*I3 - delta*R3
    dV3 = a[2]*S3 - delta*V3

    return [dS1, dI1, dR1, dV1,
            dS2, dI2, dR2, dV2,
            dS3, dI3, dR3, dV3]

# -------------------------------
# 2. Load Real Data and Fixed Parameters
# -------------------------------
start_roll = pd.Timestamp('2020-03-01')
end_roll   = pd.Timestamp('2020-04-01')

roll_data = pd.read_csv('DataSets/Korea_threeGroups_SIRV_parameter.csv')
roll_data['Date'] = pd.to_datetime(roll_data['Date'])
roll_data = roll_data[(roll_data['Date'] >= start_roll) & (roll_data['Date'] <= end_roll)].copy()
roll_data.sort_values(by='Date', inplace=True)
roll_data.reset_index(drop=True, inplace=True)
n_days = len(roll_data)
if n_days < 2:
    raise ValueError("Not enough data in the rolling period.")

def get_state_from_row(row):
    return np.array([
        row['S_0-19'], row['I_0-19'], row['R_0-19'], row['V_0-19'],
        row['S_20-49'], row['I_20-49'], row['R_20-49'], row['V_20-49'],
        row['S_50-80+'], row['I_50-80+'], row['R_50-80+'], row['V_50-80+']
    ], dtype=float)

inputs = pd.read_excel('Inputs.xlsx', header=None)
gamma = inputs.iloc[4, 0:3].values
a = inputs.iloc[10, 0:3].values
delta = inputs.iloc[8, 0]

optimal_beta_df = pd.read_csv('DataSets/Fitted_Beta_Matrix.csv', index_col=0)
optimal_beta = optimal_beta_df.values

# -------------------------------
# 3. Set Up Rolling Forecast Settings
# -------------------------------
t_span = [0, 1]  # one-day integration

# -------------------------------
# 4. Recursive Rolling Forecast (Recursive Update)
# -------------------------------
current_state = get_state_from_row(roll_data.iloc[0])
forecast_dates = []    # Dates for forecasted state (day d+1)
predicted_states = []  # Predicted states (length-12 vectors)
daily_matrices = []    # Optimal 3x3 multiplier matrix for each day

bounds = [(0, 2)] * 9
x0 = np.ones(9)

for d in range(n_days - 1):
    target_state = get_state_from_row(roll_data.iloc[d+1])
    res = minimize(lambda x: np.sum((solve_ivp(lambda t, y: sirv_model(t, y, x.reshape((3,3)) * optimal_beta, gamma, a, delta),
                                              [t_span[0], t_span[-1]], current_state,
                                              method='BDF', rtol=1e-3, atol=1e-6).y[:,-1] - target_state)**2),
                   x0, method='L-BFGS-B', bounds=bounds)
    if res.success:
        best_M = res.x.reshape((3,3))
    else:
        best_M = np.full((3,3), np.nan)
    daily_matrices.append(best_M)
    
    new_beta = best_M * optimal_beta
    sol = solve_ivp(lambda t, y: sirv_model(t, y, new_beta, gamma, a, delta),
                    [t_span[0], t_span[-1]], current_state,
                    method='BDF', rtol=1e-3, atol=1e-6)
    forecast_state = sol.y[:,-1]
    predicted_states.append(forecast_state)
    forecast_dates.append(roll_data['Date'].iloc[d+1])
    
    current_state = forecast_state.copy()
    x0 = res.x  # warm start

# -------------------------------
# 5. Extract Forecasted Infectious Data
# -------------------------------
pred_I_0_19 = [state[1] for state in predicted_states]
pred_I_20_49 = [state[5] for state in predicted_states]
pred_I_50_80 = [state[9] for state in predicted_states]
pred_I_total = [state[1] + state[5] + state[9] for state in predicted_states]

actual_I_0_19 = roll_data['I_0-19'].iloc[1:].values
actual_I_20_49 = roll_data['I_20-49'].iloc[1:].values
actual_I_50_80 = roll_data['I_50-80+'].iloc[1:].values
actual_I_total = (roll_data['I_0-19'].iloc[1:].values +
                  roll_data['I_20-49'].iloc[1:].values +
                  roll_data['I_50-80+'].iloc[1:].values)

# -------------------------------
# 6. Build DataFrame of Daily Optimal Multiplier Matrices
# -------------------------------
flat_matrices = []
for date, M in zip(roll_data['Date'].iloc[:-1], daily_matrices):
    flat_matrices.append([date] + list(M.flatten()))
cols = ['Date', 'M00', 'M01', 'M02', 'M10', 'M11', 'M12', 'M20', 'M21', 'M22']
multiplier_df = pd.DataFrame(flat_matrices, columns=cols)
multiplier_df.to_csv('DataSets/Daily_Beta_Multipliers_Rolling.csv', index=False)

# -------------------------------
# 7. Plot and Save Four Infectious Curves (Forecast vs. Actual)
# -------------------------------
import matplotlib.dates as mdates

# Define a function to save a figure.
def save_fig(fig, fname):
    fig.savefig(os.path.join(fig_folder, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)

# Plot for Age Group 0-19
fig = plt.figure(figsize=(10, 6))
plt.plot(forecast_dates, pred_I_0_19, 'ro-', label='Predicted I (0-19)')
plt.plot(forecast_dates, actual_I_0_19, 'k--', label='Actual I (0-19)')
plt.title("Recursive Forecast for Age Group 0-19")
plt.xlabel("Date")
plt.ylabel("Infectious Population")
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()
save_fig(fig, "Forecast_I_0-19.png")

# Plot for Age Group 20-49
fig = plt.figure(figsize=(10, 6))
plt.plot(forecast_dates, pred_I_20_49, 'ro-', label='Predicted I (20-49)')
plt.plot(forecast_dates, actual_I_20_49, 'k--', label='Actual I (20-49)')
plt.title("Recursive Forecast for Age Group 20-49")
plt.xlabel("Date")
plt.ylabel("Infectious Population")
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()
save_fig(fig, "Forecast_I_20-49.png")

# Plot for Age Group 50-80+
fig = plt.figure(figsize=(10, 6))
plt.plot(forecast_dates, pred_I_50_80, 'ro-', label='Predicted I (50-80+)')
plt.plot(forecast_dates, actual_I_50_80, 'k--', label='Actual I (50-80+)')
plt.title("Recursive Forecast for Age Group 50-80+")
plt.xlabel("Date")
plt.ylabel("Infectious Population")
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()
save_fig(fig, "Forecast_I_50-80+.png")

# Plot for Total Infectious Population
fig = plt.figure(figsize=(10, 6))
plt.plot(forecast_dates, pred_I_total, 'ro-', label='Predicted Total I(t)')
plt.plot(forecast_dates, actual_I_total, 'k--', label='Actual Total I(t)')
plt.title("Recursive Forecast for Total Infectious Population")
plt.xlabel("Date")
plt.ylabel("Total Infectious Population")
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()
save_fig(fig, "Forecast_Total_I.png")

# -------------------------------
# 8. Plot and Save Nine Time Series for Each Multiplier Matrix Element with Three-Level Averages
# -------------------------------

# Manually assign the subperiod boundaries
t0 = start_roll
t1 = pd.Timestamp('2020-03-03')  # Assign t1 manually
t2 = pd.Timestamp('2020-03-10')  # Assign t2 manually
t3 = end_roll

subperiods = [t0, t1, t2, t3]

for col in cols[1:]:
    fig = plt.figure(figsize=(10, 4))
    plt.plot(multiplier_df['Date'], multiplier_df[col], marker='o', linestyle='-', label=col)
    
    # Compute and plot averages for each of the three subperiods
    for i in range(3):
        t_start = subperiods[i]
        t_end = subperiods[i+1]
        # Use inclusive/exclusive bounds as needed
        if i < 2:
            mask = (multiplier_df['Date'] >= t_start) & (multiplier_df['Date'] < t_end)
        else:
            mask = (multiplier_df['Date'] >= t_start) & (multiplier_df['Date'] <= t_end)
        sub_data = multiplier_df.loc[mask, col]
        if not sub_data.empty:
            level_avg = sub_data.mean()
            plt.hlines(y=level_avg, xmin=t_start, xmax=t_end, colors='r', linestyles='--',
                       label=f'Level {i+1}: {level_avg:.3f}')
    
    plt.title(f"Time Series of {col} Over {len(multiplier_df)} Days")
    plt.xlabel("Date")
    plt.ylabel(col)
    plt.legend()
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    save_fig(fig, f"Multiplier_{col}.png")

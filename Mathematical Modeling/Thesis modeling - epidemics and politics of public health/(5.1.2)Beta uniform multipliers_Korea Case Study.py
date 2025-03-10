# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 02:52:21 2025

@author: joonc

Simulation Optimization Using Feb 26th Initial Condition
and Candidate Multiplier DataFrame

This script:
1. Loads real data and uses Feb 26 as the initial condition.
2. Loads fixed parameters from Inputs.xlsx and the fitted beta matrix.
3. Runs a simulation for a specified period starting on Feb 26.
4. Applies government intervention on Feb 27 by scaling beta using a candidate multiplier.
5. Uses minimize_scalar to find the best multiplier (with 3 decimals) that yields a simulated total peak close to a desired real-data peak.
6. Additionally, computes a DataFrame over a grid of candidate multipliers showing the simulated total peak and error.
7. Plots the simulation (total infectious) versus the restricted real data.
"""

import pandas as pd
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

# -------------------------------
# 1. Define the SIRV Model ODEs
# -------------------------------
def sirv_model(y, t, beta, gamma, a, delta):
    """
    SIRV model with three age groups.
    y = [S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3]
    beta: 3x3 transmission rate matrix
    gamma: recovery rates (array, length 3)
    a: vaccination rates (array, length 3)
    delta: waning immunity rate (scalar)
    """
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y

    # Compute populations for each group
    N1 = S1 + I1 + R1 + V1
    N2 = S2 + I2 + R2 + V2
    N3 = S3 + I3 + R3 + V3

    # Force of infection for each group
    lambda1 = sum(beta[0, j] * I / N for j, I, N in zip(range(3), [I1, I2, I3], [N1, N2, N3]))
    lambda2 = sum(beta[1, j] * I / N for j, I, N in zip(range(3), [I1, I2, I3], [N1, N2, N3]))
    lambda3 = sum(beta[2, j] * I / N for j, I, N in zip(range(3), [I1, I2, I3], [N1, N2, N3]))

    # Group 1 (e.g., 0-19)
    dS1 = -lambda1 * S1 - a[0] * S1 + delta * (R1 + V1)
    dI1 = lambda1 * S1 - gamma[0] * I1
    dR1 = gamma[0] * I1 - delta * R1
    dV1 = a[0] * S1 - delta * V1

    # Group 2 (e.g., 20-49)
    dS2 = -lambda2 * S2 - a[1] * S2 + delta * (R2 + V2)
    dI2 = lambda2 * S2 - gamma[1] * I2
    dR2 = gamma[1] * I2 - delta * R2
    dV2 = a[1] * S2 - delta * V2

    # Group 3 (e.g., 50-80+)
    dS3 = -lambda3 * S3 - a[2] * S3 + delta * (R3 + V3)
    dI3 = lambda3 * S3 - gamma[2] * I3
    dR3 = gamma[2] * I3 - delta * R3
    dV3 = a[2] * S3 - delta * V3

    # IMPORTANT: Return dR3 (not R3)
    return [dS1, dI1, dR1, dV1,
            dS2, dI2, dR2, dV2,
            dS3, dI3, dR3, dV3]

# -------------------------------
# 2. Load Data and Parameters
# -------------------------------
# (a) Load real-world data (e.g., from Korea)
data_file = 'DataSets/Korea_threeGroups_SIRV_parameter.csv'
real_data = pd.read_csv(data_file)
real_data['Date'] = pd.to_datetime(real_data['Date'])

# (b) Use Feb 26 as the initial condition date.
init_date = pd.Timestamp('2020-02-26')
row_init = real_data[real_data['Date'] == init_date]
if row_init.empty:
    raise ValueError("No data found for 2020-02-26 in the real data.")

y0 = [
    float(row_init['S_0-19'].values[0]), float(row_init['I_0-19'].values[0]),
    float(row_init['R_0-19'].values[0]), float(row_init['V_0-19'].values[0]),
    float(row_init['S_20-49'].values[0]), float(row_init['I_20-49'].values[0]),
    float(row_init['R_20-49'].values[0]), float(row_init['V_20-49'].values[0]),
    float(row_init['S_50-80+'].values[0]), float(row_init['I_50-80+'].values[0]),
    float(row_init['R_50-80+'].values[0]), float(row_init['V_50-80+'].values[0])
]

# (c) Load fixed parameters from Inputs.xlsx
inputs = pd.read_excel('Inputs.xlsx', header=None)
recovery_rates    = inputs.iloc[4, 0:3].values       # gamma (3 values)
vaccination_rates = inputs.iloc[10, 0:3].values      # a (3 values)
waning_immunity_rate = inputs.iloc[8, 0]             # delta

gamma = recovery_rates
a = vaccination_rates
delta = waning_immunity_rate

# (d) Load the fitted beta matrix (3x3) from CSV
optimal_beta_df = pd.read_csv('DataSets/Fitted_Beta_Matrix.csv', index_col=0)
optimal_beta = optimal_beta_df.values

# -------------------------------
# 3. Define Simulation Period Starting from Feb 26
# -------------------------------
extra_days = 180  # Simulation period: 90 extra days after Feb 26
t_extended = np.arange(extra_days + 1)  # 0,1,2,...,90

# Generate simulation dates starting from Feb 26
simulation_dates = [init_date + pd.Timedelta(days=int(i)) for i in t_extended]

# -------------------------------
# 4. Define Simulation with Intervention Multiplier
# -------------------------------
# Define intervention date: For example, if intervention starts on Feb 27.
intervention_date = pd.Timestamp('2020-02-27')
intervention_idx = next(i for i, d in enumerate(simulation_dates) if d >= intervention_date)

def simulate_with_multiplier(mult, y0, t_extended, intervention_idx, beta_original, gamma, a, delta):
    """
    Simulate the SIRV model starting at Feb 26 (y0).
    From the start up to intervention_idx, use the original beta.
    From intervention_idx onward, use beta scaled by 'mult'.
    Returns the full simulation result.
    """
    # Pre-intervention simulation (from day 0 to intervention)
    t_pre = t_extended[:intervention_idx+1]
    sim_pre = odeint(sirv_model, y0, t_pre, args=(beta_original, gamma, a, delta))
    
    # Post-intervention simulation (from intervention onward) using scaled beta
    beta_new = mult * beta_original
    t_post = t_extended[intervention_idx:]
    y_intervention_init = sim_pre[-1]
    sim_post = odeint(sirv_model, y_intervention_init, t_post, args=(beta_new, gamma, a, delta))
    
    # Combine results, avoiding duplication of the intervention day.
    sim_full = np.vstack((sim_pre, sim_post[1:]))
    return sim_full

def compute_total_peak(sim_result):
    """
    Compute the maximum total infectious population (sum over groups 0-19, 20-49, 50-80+).
    Infectious compartments are indices 1, 5, and 9.
    """
    I_total = sim_result[:, 1] + sim_result[:, 5] + sim_result[:, 9]
    return np.max(I_total)

# -------------------------------
# 5. Optimization: Find Best Multiplier
# -------------------------------
# Suppose the desired total peak from real data is about 4603.848.
desired_peak = 4603.848

def objective(mult, desired_peak, y0, t_extended, intervention_idx, beta_original, gamma, a, delta):
    sim_result = simulate_with_multiplier(mult, y0, t_extended, intervention_idx, beta_original, gamma, a, delta)
    model_peak = compute_total_peak(sim_result)
    return abs(model_peak - desired_peak)

result = minimize_scalar(
    objective,
    bounds=(0.0, 1.0),
    method='bounded',
    args=(desired_peak, y0, t_extended, intervention_idx, optimal_beta, gamma, a, delta)
)

if result.success:
    best_mult = result.x
    best_mult_rounded = float(f"{best_mult:.3f}")
    best_peak = compute_total_peak(simulate_with_multiplier(best_mult, y0, t_extended, intervention_idx, optimal_beta, gamma, a, delta))
    print(f"Best multiplier found: {best_mult:.6f}")
    print(f"Rounded to 3 decimals: {best_mult_rounded}")
    print(f"Simulated total peak with that multiplier: {best_peak:.2f}")
else:
    print("Optimization failed:", result.message)

# -------------------------------
# 6. Compute Candidate Multiplier Table (DataFrame)
# -------------------------------
# Here we define a grid of candidate multipliers and compute simulated peaks and error.
candidate_grid = np.linspace(0.0, 1.0, 21)  # 21 candidate multipliers from 0.0 to 1.0 (step 0.05)
peak_values = []
errors = []

for mult in candidate_grid:
    sim_result = simulate_with_multiplier(mult, y0, t_extended, intervention_idx, optimal_beta, gamma, a, delta)
    peak = compute_total_peak(sim_result)
    error = abs(peak - desired_peak)
    peak_values.append(peak)
    errors.append(error)

multiplier_df = pd.DataFrame({
    "Multiplier": candidate_grid,
    "Simulated Total Peak": peak_values,
    "Absolute Error": errors
})

print("\nCandidate Multiplier Table:")
print(multiplier_df)

# Optionally, save the DataFrame to CSV:
multiplier_df.to_csv('DataSets/Candidate_Multiplier_Table.csv', index=False)

# -------------------------------
# 7. Plot Simulation vs. Real Data (Total Infectious)
# -------------------------------
# We'll plot the simulation for a few candidate multipliers, including the best one.
plot_candidates = [0.3, 0.2, best_mult_rounded]
simulations = {}
for mult in plot_candidates:
    simulations[mult] = simulate_with_multiplier(mult, y0, t_extended, intervention_idx, optimal_beta, gamma, a, delta)

plt.figure(figsize=(10, 6))
for mult, sim in simulations.items():
    I_total_sim = sim[:, 1] + sim[:, 5] + sim[:, 9]
    plt.plot(simulation_dates, I_total_sim, label=f'β Multiplier = {mult}')
    
# Restrict real data to simulation domain (from Feb 26 to Feb26+extra_days)
real_plot = real_data[(real_data['Date'] >= simulation_dates[0]) & (real_data['Date'] <= simulation_dates[-1])]
I_real_total = (real_plot['I_0-19'].values + real_plot['I_20-49'].values + real_plot['I_50-80+'].values)
plt.plot(real_plot['Date'], I_real_total, 'k--', label='Real Total I(t)')

plt.axvline(x=intervention_date, color='gray', linestyle=':', label='Intervention (Feb 27)')
plt.title("Total Infectious Population (Simulation vs. Real Data)")
plt.xlabel("Date")
plt.ylabel("Total Infectious Population")
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.xlim([simulation_dates[0], simulation_dates[-1]])
plt.show()


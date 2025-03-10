# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 03:15:23 2025

@author: joonc
"""

# -*- coding: utf-8 -*-
"""
Parameter Fitting: β-Multiplier via Full Trajectory Fitting
Using Feb 26 as the initial condition and minimizing SSE over an extended period.
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
    gamma: recovery rates (length 3)
    a: vaccination rates (length 3)
    delta: waning immunity rate (scalar)
    """
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y
    
    # Population sizes per group
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
    
    # IMPORTANT: Return dR3 (not R3)
    return [dS1, dI1, dR1, dV1,
            dS2, dI2, dR2, dV2,
            dS3, dI3, dR3, dV3]

# -------------------------------
# 2. Load Data and Parameters
# -------------------------------
# (a) Load full real-world data (Korea dataset)
data_file = 'DataSets/Korea_threeGroups_SIRV_parameter.csv'
real_data = pd.read_csv(data_file)
real_data['Date'] = pd.to_datetime(real_data['Date'])

# (b) Use Feb 26 as the initial condition (ensure a row for '2020-02-26' exists)
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
extra_days = 10  # For example, extend simulation 260 days from Feb 26.
t_extended = np.arange(extra_days + 1)  # Days: 0, 1, 2, ..., extra_days

# Generate simulation dates starting from Feb 26
simulation_dates = [init_date + pd.Timedelta(days=int(i)) for i in t_extended]

# -------------------------------
# 4. Define Simulation with Intervention Multiplier
# -------------------------------
# Assume government intervention starts on Feb 27
intervention_date = pd.Timestamp('2020-02-27')
intervention_idx = next(i for i, d in enumerate(simulation_dates) if d >= intervention_date)

def simulate_with_multiplier(mult, y0, t_extended, intervention_idx, beta_original, gamma, a, delta):
    """
    Simulate the SIRV model starting on Feb 26.
    Use the original beta up to the intervention date, then use beta scaled by 'mult'.
    Returns the full simulation result.
    """
    t_pre = t_extended[:intervention_idx+1]
    sim_pre = odeint(sirv_model, y0, t_pre, args=(beta_original, gamma, a, delta))
    beta_new = mult * beta_original
    t_post = t_extended[intervention_idx:]
    y_intervention_init = sim_pre[-1]
    sim_post = odeint(sirv_model, y_intervention_init, t_post, args=(beta_new, gamma, a, delta))
    sim_full = np.vstack((sim_pre, sim_post[1:]))
    return sim_full

def compute_total_peak(sim_result):
    """
    Compute the maximum total infected over the simulation.
    Infectious compartments are indices 1, 5, and 9.
    """
    I_total = sim_result[:, 1] + sim_result[:, 5] + sim_result[:, 9]
    return np.max(I_total)

# -------------------------------
# 5. Define Objective Function Over Full Period
# -------------------------------
# Instead of matching a single peak value, we now fit the beta multiplier
# by minimizing the sum of squared errors (SSE) between the simulated total infected
# and the real total infected over the entire simulation period.

# First, restrict real data to the simulation domain.
real_sim_data = real_data[(real_data['Date'] >= simulation_dates[0]) & (real_data['Date'] <= simulation_dates[-1])].copy()
# For total infected from real data:
real_total = (real_sim_data['I_0-19'].values + 
              real_sim_data['I_20-49'].values + 
              real_sim_data['I_50-80+'].values)

def objective(mult, y0, t_extended, intervention_idx, beta_original, gamma, a, delta, real_total):
    sim_result = simulate_with_multiplier(mult, y0, t_extended, intervention_idx, beta_original, gamma, a, delta)
    I_sim = sim_result[:, 1] + sim_result[:, 5] + sim_result[:, 9]
    error = np.sum((I_sim - real_total)**2)
    return error

# -------------------------------
# 6. Optimize Beta Multiplier Over Full Period
# -------------------------------
result = minimize_scalar(
    objective,
    bounds=(0.0, 1.0),
    method='bounded',
    args=(y0, t_extended, intervention_idx, optimal_beta, gamma, a, delta, real_total)
)

if result.success:
    best_mult = result.x
    best_mult_precise = best_mult  # Here, we can display many decimals.
    best_mult_rounded = float(f"{best_mult:.7f}")  # Round to 7 decimals
    best_error = result.fun
    sim_best = simulate_with_multiplier(best_mult, y0, t_extended, intervention_idx, optimal_beta, gamma, a, delta)
    best_peak = compute_total_peak(sim_best)
    print(f"Best multiplier found (full trajectory fit): {best_mult_precise:.10f}")
    print(f"Rounded to 7 decimals: {best_mult_rounded}")
    print(f"Simulated total peak with that multiplier: {best_peak:.2f}")
    print(f"Objective (SSE) value: {best_error:.2f}")
else:
    print("Optimization failed:", result.message)

# -------------------------------
# 7. Build a DataFrame of Candidate Multipliers
# -------------------------------
candidate_grid = np.linspace(0.0, 1.0, 51)  # 51 candidates from 0.0 to 1.0
peak_values = []
sse_values = []

for mult in candidate_grid:
    sim_result = simulate_with_multiplier(mult, y0, t_extended, intervention_idx, optimal_beta, gamma, a, delta)
    peak = compute_total_peak(sim_result)
    sse = objective(mult, y0, t_extended, intervention_idx, optimal_beta, gamma, a, delta, real_total)
    peak_values.append(peak)
    sse_values.append(sse)

multiplier_df = pd.DataFrame({
    "Multiplier": candidate_grid,
    "Simulated Total Peak": peak_values,
    "SSE": sse_values
})

print("\nCandidate Multiplier Table (Full Trajectory Fitting):")
print(multiplier_df)

# Save the DataFrame to CSV
multiplier_df.to_csv('DataSets/Candidate_Multiplier_FullTrajectory.csv', index=False)

# -------------------------------
# 8. Plot Simulation vs. Real Data (Total Infected)
# -------------------------------
# Plot the simulation for the best multiplier and some candidate values.
plot_candidates = [0.3, 0.2, best_mult_rounded]
simulations = {}
for mult in plot_candidates:
    simulations[mult] = simulate_with_multiplier(mult, y0, t_extended, intervention_idx, optimal_beta, gamma, a, delta)

plt.figure(figsize=(10, 6))
for mult, sim in simulations.items():
    I_total_sim = sim[:, 1] + sim[:, 5] + sim[:, 9]
    plt.plot(simulation_dates, I_total_sim, label=f'β Multiplier = {mult}')
    
# Plot the real data (restricted to the simulation domain)
plt.plot(real_sim_data['Date'], real_total, 'k--', label='Real Total I(t)')

plt.axvline(x=intervention_date, color='gray', linestyle=':', label='Intervention (Feb 27)')
plt.title("Total Infectious Population: Simulation vs. Real Data")
plt.xlabel("Date")
plt.ylabel("Total Infectious Population")
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.xlim([simulation_dates[0], simulation_dates[-1]])
plt.show()

# Run simulation with the best multiplier found
sim_best = simulate_with_multiplier(best_mult, y0, t_extended, intervention_idx, optimal_beta, gamma, a, delta)

# Restrict real data to the simulation domain (from initial date to the last simulation date)
real_plot = real_data[(real_data['Date'] >= simulation_dates[0]) &
                      (real_data['Date'] <= simulation_dates[-1])].copy()

# 1. Plot for Age Group 0-19
plt.figure(figsize=(10, 6))
plt.plot(simulation_dates, sim_best[:, 1], 'r-', label='Simulated I (0-19)')
plt.plot(real_plot['Date'], real_plot['I_0-19'], 'k--', label='Real I (0-19)')
plt.axvline(x=intervention_date, color='gray', linestyle=':', label='Intervention (Feb 27)')
plt.title("Infectious Population for Age Group 0-19")
plt.xlabel("Date")
plt.ylabel("Infectious Population")
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.xlim([simulation_dates[0], simulation_dates[-1]])
plt.show()

# 2. Plot for Age Group 20-49
plt.figure(figsize=(10, 6))
plt.plot(simulation_dates, sim_best[:, 5], 'r-', label='Simulated I (20-49)')
plt.plot(real_plot['Date'], real_plot['I_20-49'], 'k--', label='Real I (20-49)')
plt.axvline(x=intervention_date, color='gray', linestyle=':', label='Intervention (Feb 27)')
plt.title("Infectious Population for Age Group 20-49")
plt.xlabel("Date")
plt.ylabel("Infectious Population")
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.xlim([simulation_dates[0], simulation_dates[-1]])
plt.show()

# 3. Plot for Age Group 50-80+
plt.figure(figsize=(10, 6))
plt.plot(simulation_dates, sim_best[:, 9], 'r-', label='Simulated I (50-80+)')
plt.plot(real_plot['Date'], real_plot['I_50-80+'], 'k--', label='Real I (50-80+)')
plt.axvline(x=intervention_date, color='gray', linestyle=':', label='Intervention (Feb 27)')
plt.title("Infectious Population for Age Group 50-80+")
plt.xlabel("Date")
plt.ylabel("Infectious Population")
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.xlim([simulation_dates[0], simulation_dates[-1]])
plt.show()

# 4. Plot for Total Infectious Population (sum over all groups)
I_total_sim = sim_best[:, 1] + sim_best[:, 5] + sim_best[:, 9]
I_total_real = (real_plot['I_0-19'].values +
                real_plot['I_20-49'].values +
                real_plot['I_50-80+'].values)

plt.figure(figsize=(10, 6))
plt.plot(simulation_dates, I_total_sim, 'r-', label='Simulated Total I(t)')
plt.plot(real_plot['Date'], I_total_real, 'k--', label='Real Total I(t)')
plt.axvline(x=intervention_date, color='gray', linestyle=':', label='Intervention (Feb 27)')
plt.title("Total Infectious Population (All Age Groups)")
plt.xlabel("Date")
plt.ylabel("Total Infectious Population")
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.xlim([simulation_dates[0], simulation_dates[-1]])
plt.show()

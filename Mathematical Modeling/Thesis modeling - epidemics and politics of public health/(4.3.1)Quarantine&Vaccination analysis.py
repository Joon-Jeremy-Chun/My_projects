# -*- coding: utf-8 -*-
"""
Created on Mon Feb 9 10:16:04 2025

@author: joonc

Extended Model with Dynamic Vaccination Strategy (chapter3.5) plus Scenario Analysis

Two simulation scenarios are considered:
    1. Social Distancing Only: Apply a uniform percentage reduction to the β matrix.
    2. Vaccination Only: Vary the constant vaccination rate (manual option) with no modification to β.

For each simulation, we record the peak number of infections (total across groups)
and the day on which that peak occurs.
"""

import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# -----------------------------
# 1. LOAD PARAMETERS AND DATA
# -----------------------------
def load_parameters(file_path):
    df = pd.read_excel(file_path, header=None)
    
    recovery_rates = df.iloc[4, 0:3].values
    maturity_rates = df.iloc[6, 0:2].values
    waning_immunity_rate = df.iloc[8, 0]
    time_span = df.iloc[12, 0]
    population_size = df.iloc[14, 0:3].values
    susceptible_init = df.iloc[14, 0:3].values
    infectious_init = df.iloc[16, 0:3].values  
    recovered_init = df.iloc[18, 0:3].values   
    vaccinated_init = df.iloc[20, 0:3].values  
    
    return (recovery_rates, maturity_rates, waning_immunity_rate,
             time_span, population_size, susceptible_init,
             infectious_init, recovered_init, vaccinated_init)

# Load parameters from Excel file
file_path = 'Inputs.xlsx'
(gamma, mu, W, time_span, N, S_init, I_init, R_init, V_init) = load_parameters(file_path)

# Load transmission rates matrix from CSV file
transmission_rates_csv_path = 'DataSets/Fitted_Beta_Matrix.csv'
beta = pd.read_csv(transmission_rates_csv_path).iloc[0:3, 1:4].values

# Define initial conditions for 3 groups (Children, Adults, Seniors)
initial_conditions = [
    S_init[0], I_init[0], R_init[0], V_init[0],  
    S_init[1], I_init[1], R_init[1], V_init[1],  
    S_init[2], I_init[2], R_init[2], V_init[2]   
]

# -----------------------------
# 2. VACCINATION STRATEGY SETUP
# -----------------------------
# Option 1: Manual constant vaccination rate (used in Vaccination Only scenario)
vaccination_rates = np.array([0.02, 0.02, 0.02])  # default manual rate
vaccination_rates_dynamic = vaccination_rates #op1


# Option 2: Dynamic vaccination rates based on target proportions
k1, d1 = 0.8, 30  # % in  days for Children
k2, d2 = 0.7, 45  # % in  days for Adults
k3, d3 = 0.9, 60  # % in  days for Seniors

# Calculate daily vaccination rate for each group (dynamic)
x1 = k1 ** (1/d1)
x2 = k2 ** (1/d2)
x3 = k3 ** (1/d3)
#vaccination_rates_dynamic = np.array([x1, x2, x3]) #op2

# Stage times and boost factors (for vaccination strategy phases)
t1, t2 = 30, 60
B_timeP0, B_timeP1, B_timeP2 = 1.0, 1.0, 1.0

def vaccination_strategy(t, use_dynamic=True):
    """
    Returns the vaccination rate vector at time t.
    If use_dynamic is True, use the dynamic rates; otherwise, use the manual rates.
    """
    rates = vaccination_rates_dynamic if use_dynamic else vaccination_rates
    if t < t1:
        return rates * B_timeP0  # Initial phase
    elif t1 <= t < t2:
        return rates * B_timeP1  # Full-scale phase
    else:
        return rates * B_timeP2  # Booster phase

# -----------------------------
# 3. MODEL DEFINITION
# -----------------------------
def deriv(y, t, N, beta, gamma, mu, W, use_dynamic):
    """
    Extended SIRV model ODEs with three population groups.
    
    use_dynamic: Boolean indicating whether to use dynamic or manual vaccination rates.
    """
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y
    a_t = vaccination_strategy(t, use_dynamic)
    
    # Force of infection for each group
    lambda1 = beta[0, 0] * I1/N[0] + beta[0, 1] * I2/N[1] + beta[0, 2] * I3/N[2]
    lambda2 = beta[1, 0] * I1/N[0] + beta[1, 1] * I2/N[1] + beta[1, 2] * I3/N[2]
    lambda3 = beta[2, 0] * I1/N[0] + beta[2, 1] * I2/N[1] + beta[2, 2] * I3/N[2]
    
    # Group 1 (Children)
    dS1dt = -lambda1 * S1 - a_t[0] * S1 + W * R1 + W * V1
    dI1dt = lambda1 * S1 - gamma[0] * I1 - mu[0] * I1
    dR1dt = gamma[0] * I1 - W * R1 - mu[0] * R1
    dV1dt = a_t[0] * S1 - W * V1
    
    # Group 2 (Adults)
    dS2dt = -lambda2 * S2 + mu[0] * S1 - a_t[1] * S2 + W * R2 + W * V2
    dI2dt = lambda2 * S2 + mu[0] * I1 - gamma[1] * I2 - mu[1] * I2
    dR2dt = gamma[1] * I2 + mu[0] * R1 - W * R2 - mu[1] * R2
    dV2dt = a_t[1] * S2 - W * V2
    
    # Group 3 (Seniors)
    dS3dt = -lambda3 * S3 + mu[1] * S2 - a_t[2] * S3 + W * R3 + W * V3
    dI3dt = lambda3 * S3 + mu[1] * I2 - gamma[2] * I3
    dR3dt = gamma[2] * I3 + mu[1] * R2 - W * R3
    dV3dt = a_t[2] * S3 - W * V3
    
    return [dS1dt, dI1dt, dR1dt, dV1dt,
            dS2dt, dI2dt, dR2dt, dV2dt,
            dS3dt, dI3dt, dR3dt, dV3dt]

# -----------------------------
# 4. SIMULATION FUNCTION
# -----------------------------
def run_simulation(beta_mod, use_dynamic):
    """
    Runs the model simulation using the given beta matrix (beta_mod) and vaccination strategy.
    Returns the peak total infections and the day of the peak.
    """
    t = np.linspace(0, time_span, int(time_span))
    # Solve the ODEs; note we now pass use_dynamic to the derivative function
    results = odeint(deriv, initial_conditions, t, args=(N, beta_mod, gamma, mu, W, use_dynamic))
    
    # Compute total infections across the three groups
    I_total = results[:, 1] + results[:, 5] + results[:, 9]
    peak_idx = np.argmax(I_total)
    peak_value = I_total[peak_idx]
    peak_day = t[peak_idx]
    return peak_value, peak_day

# -----------------------------
# 5. SCENARIO 1: SOCIAL DISTANCING ONLY
# -----------------------------
social_results = []
# Define a range of multipliers for β (e.g., from no reduction (1.0) to a 50% reduction (0.5))
beta_multipliers = np.linspace(1.0, 0.5, num=6)  # 1.0, 0.9, 0.8, 0.7, 0.6, 0.5

for multiplier in beta_multipliers:
    beta_mod = beta * multiplier  # apply uniform reduction
    # Use dynamic vaccination (i.e., no change in vaccination rates) for this scenario
    peak_value, peak_day = run_simulation(beta_mod, use_dynamic=True)
    social_results.append({
        'beta_multiplier': multiplier,
        'peak_infected': peak_value,
        'peak_day': peak_day
    })

social_df = pd.DataFrame(social_results)
print("Social Distancing Scenario:")
print(social_df)

# -----------------------------
# 6. SCENARIO 2: VACCINATION ONLY
# -----------------------------
vaccination_results = []
# For this scenario, we keep the original β matrix (no social distancing) and vary the vaccination rate.
# We will override the manual constant vaccination_rates and force use_dynamic=False.
vaccination_rates_range = np.linspace(0.02, 0.1, num=9)  # from 0.02 to 0.1

for v in vaccination_rates_range:
    # Override the manual vaccination rate (global variable used when use_dynamic=False)
    vaccination_rates = np.array([v, v, v])
    # Run simulation with the original beta matrix and manual vaccination strategy
    peak_value, peak_day = run_simulation(beta, use_dynamic=False)
    vaccination_results.append({
        'vaccination_rate': v,
        'peak_infected': peak_value,
        'peak_day': peak_day
    })

vaccination_df = pd.DataFrame(vaccination_results)
print("\nVaccination Scenario:")
print(vaccination_df)

# -----------------------------
# 7. OPTIONAL: PLOTTING EXAMPLE (for one simulation)
# -----------------------------
# Here is an example of how you might plot the infection curve for one case (e.g., no interventions)
t = np.linspace(0, time_span, int(time_span))
results = odeint(deriv, initial_conditions, t, args=(N, beta, gamma, mu, W, True))
I_children = results[:, 1]
I_adults = results[:, 5]
I_seniors = results[:, 9]
I_total = I_children + I_adults + I_seniors

plt.figure(figsize=(10, 6))
plt.plot(t, I_total, label='Total Infected')
peak_idx = np.argmax(I_total)
plt.scatter(t[peak_idx], I_total[peak_idx], color='red',
            label=f'Peak at day {t[peak_idx]:.2f}\n({I_total[peak_idx]:.2f} infections)')
plt.xlabel('Days')
plt.ylabel('Number of Infected Individuals')
plt.title('Total Infections Over Time (No Intervention Modification)')
plt.legend()
plt.grid()
plt.show()


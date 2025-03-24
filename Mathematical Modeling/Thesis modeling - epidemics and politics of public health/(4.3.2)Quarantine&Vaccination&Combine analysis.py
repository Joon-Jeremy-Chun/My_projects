# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 10:20:14 2025

@author: joonc

Extended Model with Dynamic Vaccination Strategy (Chapter 3.5) plus Scenario Analysis

Three simulation scenarios are considered:
    1. Social Distancing Only: Apply a uniform percentage reduction to the β matrix.
    2. Vaccination Only: Vary the constant vaccination rate (manual option) with no modification to β.
    3. Vaccination with Light Social Distancing: Apply a moderate reduction to β and vary the manual vaccination rate.

For each simulation, we record the peak number of infections (total across groups)
and the day on which that peak occurs. In addition, for Scenarios 2 and 3 we compute:
    - the total number of vaccinated individuals and the ratio (vaccinated/total) at day 30,
    - at day 60, and
    - at day 365.
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
# Option 1: Manual constant vaccination rate (used in Vaccination Only and combined scenario)
vaccination_rates = np.array([0.0, 0.0, 0.0])  # default manual rate
vaccination_rates_dynamic = vaccination_rates

# Option 2: Dynamic vaccination rates based on target proportions (not used in these scenarios)
k1, d1 = 0.8, 30  # e.g., 80% in 30 days for Children
k2, d2 = 0.7, 45  # e.g., 70% in 45 days for Adults
k3, d3 = 0.9, 60  # e.g., 90% in 60 days for Seniors
x1 = k1 ** (1/d1)
x2 = k2 ** (1/d2)
x3 = k3 ** (1/d3)
# vaccination_rates_dynamic = np.array([x1, x2, x3])
# In our current scenarios, we use the manual vaccination rate (set use_dynamic=False).

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
    Returns:
      - peak_value: peak total infections,
      - peak_day: the day of the peak,
      - total_vaccinated_30, ratio_vaccinated_30 at day 30,
      - total_vaccinated_60, ratio_vaccinated_60 at day 60,
      - total_vaccinated_365, ratio_vaccinated_365 at day 365.
    """
    t = np.linspace(0, time_span, int(time_span))
    results = odeint(deriv, initial_conditions, t, args=(N, beta_mod, gamma, mu, W, use_dynamic))
    
    # Compute total infections across the three groups
    I_total = results[:, 1] + results[:, 5] + results[:, 9]
    peak_idx = np.argmax(I_total)
    peak_value = I_total[peak_idx]
    peak_day = t[peak_idx]
    
    # Determine indices corresponding to day 30, 60, and 365 (if available)
    index_30 = np.where(t >= 30)[0][0]
    total_vaccinated_30 = results[index_30, 3] + results[index_30, 7] + results[index_30, 11]
    ratio_vaccinated_30 = total_vaccinated_30 / np.sum(N)
    
    index_60 = np.where(t >= 60)[0][0]
    total_vaccinated_60 = results[index_60, 3] + results[index_60, 7] + results[index_60, 11]
    ratio_vaccinated_60 = total_vaccinated_60 / np.sum(N)
    
    # For day 365, check if simulation ran that long; otherwise use last index.
    if t[-1] >= 365:
        index_365 = np.where(t >= 365)[0][0]
    else:
        index_365 = -1  # last index
    total_vaccinated_365 = results[index_365, 3] + results[index_365, 7] + results[index_365, 11]
    ratio_vaccinated_365 = total_vaccinated_365 / np.sum(N)
    
    return (peak_value, peak_day, 
            total_vaccinated_30, ratio_vaccinated_30,
            total_vaccinated_60, ratio_vaccinated_60,
            total_vaccinated_365, ratio_vaccinated_365)

# -----------------------------
# 5. SCENARIO 1: SOCIAL DISTANCING ONLY
# -----------------------------
social_results = []
beta_multipliers = [1.0, 0.7, 0.35, 0.2]
for multiplier in beta_multipliers:
    beta_mod = beta * multiplier  # apply uniform reduction
    # Use dynamic vaccination (vaccination strategy unchanged)
    peak_value, peak_day, _, _, _, _, _, _ = run_simulation(beta_mod, use_dynamic=True)
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
vaccination_rates_range = np.linspace(0.0001, 0.0201, num=201)
#vaccination_rates_range = np.linspace(0.001, 0.401, num=401)
for v in vaccination_rates_range:
    # Override the manual vaccination rate (global variable used when use_dynamic=False)
    vaccination_rates = np.array([v, v, v])
    (peak_value, peak_day, 
     total_vacc_30, ratio_vacc_30,
     total_vacc_60, ratio_vacc_60,
     total_vacc_365, ratio_vacc_365) = run_simulation(beta, use_dynamic=False)
    vaccination_results.append({
        'vaccination_rate': v,
        'peak_infected': peak_value,
        'peak_day': peak_day,
        'total_vaccinated_30': total_vacc_30,
        'ratio_vaccinated_30': ratio_vacc_30,
        'total_vaccinated_60': total_vacc_60,
        'ratio_vaccinated_60': ratio_vacc_60,
        'total_vaccinated_365': total_vacc_365,
        'ratio_vaccinated_365': ratio_vacc_365
    })

vaccination_df = pd.DataFrame(vaccination_results)
print("\nVaccination Only Scenario:")
print(vaccination_df)

# -----------------------------
# 7. SCENARIO 3: VACCINATION WITH LIGHT SOCIAL DISTANCING
# -----------------------------
combined_results = []
beta_multiplier_combined = 0.7
beta_mod_combined = beta * beta_multiplier_combined
for v in vaccination_rates_range:
    vaccination_rates = np.array([v, v, v])
    (peak_value, peak_day, 
     total_vacc_30, ratio_vacc_30,
     total_vacc_60, ratio_vacc_60,
     total_vacc_365, ratio_vacc_365) = run_simulation(beta_mod_combined, use_dynamic=False)
    combined_results.append({
        'vaccination_rate': v,
        'beta_multiplier': beta_multiplier_combined,
        'peak_infected': peak_value,
        'peak_day': peak_day,
        'total_vaccinated_30': total_vacc_30,
        'ratio_vaccinated_30': ratio_vacc_30,
        'total_vaccinated_60': total_vacc_60,
        'ratio_vaccinated_60': ratio_vacc_60,
        'total_vaccinated_365': total_vacc_365,
        'ratio_vaccinated_365': ratio_vacc_365
    })

combined_df = pd.DataFrame(combined_results)
print("\nVaccination with Light Social Distancing Scenario:")
print(combined_df)

# -----------------------------
# 8. OPTIONAL: PLOTTING EXAMPLE (for one simulation)
# -----------------------------
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
plt.title('Total Infections Over Time (Baseline)')
plt.legend()
plt.grid()
plt.show()

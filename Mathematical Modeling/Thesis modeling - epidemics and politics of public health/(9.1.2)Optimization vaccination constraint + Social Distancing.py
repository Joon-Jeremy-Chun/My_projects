# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 00:26:32 2025

@author: joonc

Comparison of Two Scenarios:
1. Vaccination Only (β multiplier = 1.0)
2. Vaccination with Light Social Distancing (β multiplier = 0.7)

This code uses the previously defined functions:
- deriv: the SIRV model with constant vaccination rates.
- instant_cost: to compute the instantaneous economic cost.
- simulate_with_cost: to run the simulation over defined periods and accumulate cost.

The vaccination constraint is enforced with:
   0.1207*V1 + 0.6092*V2 + 0.2700*V3 = 0.01067

Tentative vaccination rates are used for V1, V2, and V3.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# -----------------------------
# 1. PARAMETERS (same as before)
# -----------------------------
# Epidemic parameters (example values)
M_mild = 160.8                     # daily medical cost per infected (mild)
daily_income = np.array([27.74, 105.12, 92.45])  # children, adults, seniors
employment_rates = np.array([0.0, 0.694, 0.513])  # assume children not employed
funeral_cost = 9405.38
GDP_per_capita = 31929
# GDP loss exponential model parameters (from previous code)
a_param, b_param, c_param = -0.3648, -8.7989, -0.0012

# Population sizes for three groups: Children, Adults, Seniors
populations = np.array([6246073, 31530507, 13972160])
total_population = np.sum(populations)
# Normalized population ratios (for vaccination constraint)
r = np.array([0.1207, 0.6092, 0.2700])

# Vaccination constraint:
# r[0]*V1 + r[1]*V2 + r[2]*V3 = 0.01067.
# Tentative values (one possible set):
V1 = 0.001      # for children
V2 = 0.005      # for adults
V3 = (0.01067 - r[0]*V1 - r[1]*V2) / r[2]  # solved for seniors
print("Tentative vaccination rates:", V1, V2, V3)

# Set the constant vaccination rates vector
V_const = np.array([V1, V2, V3])

# Other epidemiological parameters
gamma = np.array([0.125, 0.083, 0.048])  # recovery rates for three groups
W = 0.005                              # waning immunity rate

# Transmission matrix (example fitted matrix)
beta = np.array([[0.1901, 0.0660, 0.0028],
                 [0.1084, 0.1550, 0.0453],
                 [0.1356, 0.1540, 0.1165]])

# -----------------------------
# 2. TIME SETTINGS
# -----------------------------
# Define time periods: 0, 30, 60, 90, 120, 150 days
t_periods = [0, 30, 60, 90, 120, 150]
dt = 1.0  # time step in days

# -----------------------------
# 3. MODEL DEFINITION (SIRV with Vaccination)
# -----------------------------
def deriv(y, t, N, beta, gamma, W, V_const):
    """
    Extended SIRV model with constant vaccination rates V_const = [V1, V2, V3].
    y: vector [S1, I1, R1, V1_state, S2, I2, R2, V2_state, S3, I3, R3, V3_state]
    """
    S1, I1, R1, V1_state, S2, I2, R2, V2_state, S3, I3, R3, V3_state = y
    # Force of infection for each group
    lambda1 = beta[0, 0]*I1/populations[0] + beta[0, 1]*I2/populations[1] + beta[0, 2]*I3/populations[2]
    lambda2 = beta[1, 0]*I1/populations[0] + beta[1, 1]*I2/populations[1] + beta[1, 2]*I3/populations[2]
    lambda3 = beta[2, 0]*I1/populations[0] + beta[2, 1]*I2/populations[1] + beta[2, 2]*I3/populations[2]
    
    # Group 1 (Children)
    dS1dt = -lambda1 * S1 - V_const[0]*S1 + W*(R1 + V1_state)
    dI1dt = lambda1 * S1 - gamma[0]*I1
    dR1dt = gamma[0]*I1 - W*R1
    dV1dt = V_const[0]*S1 - W*V1_state

    # Group 2 (Adults)
    dS2dt = -lambda2 * S2 - V_const[1]*S2 + W*(R2 + V2_state)
    dI2dt = lambda2 * S2 - gamma[1]*I2
    dR2dt = gamma[1]*I2 - W*R2
    dV2dt = V_const[1]*S2 - W*V2_state

    # Group 3 (Seniors)
    dS3dt = -lambda3 * S3 - V_const[2]*S3 + W*(R3 + V3_state)
    dI3dt = lambda3 * S3 - gamma[2]*I3
    dR3dt = gamma[2]*I3 - W*R3
    dV3dt = V_const[2]*S3 - W*V3_state

    return [dS1dt, dI1dt, dR1dt, dV1dt,
            dS2dt, dI2dt, dR2dt, dV2dt,
            dS3dt, dI3dt, dR3dt, dV3dt]

# -----------------------------
# 4. COST FUNCTION (Instantaneous Cost Rate)
# -----------------------------
def instant_cost(y, beta_m):
    """
    Computes the instantaneous cost rate (per day) given the state vector y.
    y: state vector from the SIRV model.
    beta_m: current beta multiplier (used to compute GDP loss rate).
    """
    S1, I1, R1, V1_state, S2, I2, R2, V2_state, S3, I3, R3, V3_state = y
    I_total = I1 + I2 + I3
    # Medical cost rate
    med_rate = M_mild * I_total
    # Wage loss: for adults and seniors plus caregiving cost for children (using adult wage)
    wage_rate = daily_income[1]*employment_rates[1]*(I2 + I1) + daily_income[2]*employment_rates[2]*I3
    # Vaccination cost rate: cost per dose times daily vaccinations = V_const * S for each group
    vaccinations = V_const[0]*S1 + V_const[1]*S2 + V_const[2]*S3
    vacc_rate = vaccination_cost_per_capita * vaccinations
    # GDP loss rate: scale annual GDP loss fraction by 1/365 (using beta_m for intervention intensity)
    GDP_loss_fraction = a_param * np.exp(b_param * beta_m) + c_param
    gdp_rate = GDP_per_capita * abs(GDP_loss_fraction) * (total_population / 365)
    
    total_rate = med_rate + wage_rate + vacc_rate + gdp_rate
    return total_rate

# Define vaccination cost per capita (needed in instant_cost)
vaccination_cost_per_capita = 35.76

# -----------------------------
# 5. SIMULATION WITH COST ACCUMULATION OVER PERIODS
# -----------------------------
def simulate_with_cost(V_const, beta_mod, t_periods, populations, gamma, W):
    """
    Runs the SIRV model over the defined periods using constant vaccination rates V_const.
    Returns arrays for time and cumulative cost.
    """
    # Initial conditions: assume all are susceptible except a small initial infection in each group.
    S0 = populations.copy()
    I0 = np.array([10, 10, 10])  # small initial infections
    R0 = np.zeros(3)
    V0 = np.zeros(3)
    y0 = [S0[0], I0[0], R0[0], V0[0],
          S0[1], I0[1], R0[1], V0[1],
          S0[2], I0[2], R0[2], V0[2]]
    
    cumulative_cost = []
    t_cum = []
    total_cost = 0.0
    
    current_state = y0
    for i in range(len(t_periods)-1):
        t_start = t_periods[i]
        t_end = t_periods[i+1]
        t_span = np.arange(t_start, t_end+dt, dt)
        # For this simulation, beta multiplier is applied in beta_mod already.
        # For GDP cost, we assume beta multiplier = multiplier (here extracted from beta_mod)
        # (Assume uniform multiplier across all elements; extract first element divided by original beta[0,0])
        beta_multiplier = beta_mod[0,0] / beta[0,0]
        
        sol = odeint(deriv, current_state, t_span, args=(populations, beta, gamma, W, V_const))
        
        for t_val, y in zip(t_span, sol):
            cost_rate = instant_cost(y, beta_multiplier)
            total_cost += cost_rate * dt  # dt = 1 day
            t_cum.append(t_val)
            cumulative_cost.append(total_cost)
        
        current_state = sol[-1]
    
    return np.array(t_cum), np.array(cumulative_cost)

# -----------------------------
# 6. RUN TWO SCENARIOS AND COMPARE
# -----------------------------
# Scenario 1: Vaccination Only (no additional social distancing; beta multiplier = 1.0)
beta_mod1 = beta * 1.0
t_cum1, cum_cost1 = simulate_with_cost(V_const, beta_mod1, t_periods, populations, gamma, W)

# Scenario 2: Vaccination with Light Social Distancing (beta multiplier = 0.7)
beta_mod2 = beta * 0.7
t_cum2, cum_cost2 = simulate_with_cost(V_const, beta_mod2, t_periods, populations, gamma, W)

# -----------------------------
# 7. VISUALIZATION: CUMULATIVE COST VS. TIME
# -----------------------------
plt.figure(figsize=(10, 6))
plt.plot(t_cum1, cum_cost1, label='Vaccination Only (β multiplier = 1.0)', marker='o', linestyle='-')
plt.plot(t_cum2, cum_cost2, label='Vaccination with Light Social Distancing (β multiplier = 0.7)', marker='s', linestyle='--')
plt.xlabel('Time (days)')
plt.ylabel('Cumulative Economic Cost (USD)')
plt.title('Cumulative Economic Cost Comparison Over 150 Days')
plt.legend()
plt.grid(True)
plt.show()

# Print final total costs for each scenario
print("Total Economic Cost over 150 days (Vaccination Only): ${:,.2f}".format(cum_cost1[-1]))
print("Total Economic Cost over 150 days (Vaccination with Light Social Distancing): ${:,.2f}".format(cum_cost2[-1]))


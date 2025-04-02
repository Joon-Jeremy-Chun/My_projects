# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 23:44:36 2025


@author: joonc

Dynamic Vaccination Cost Optimization Model:
- Simulation over 150 days divided into five 30-day periods.
- Constant vaccination rates V1, V2, V3 are applied over all periods.
- Vaccination constraint: 0.1207*V1 + 0.6092*V2 + 0.2700*V3 = 0.01067.
- Computes cumulative economic cost over time.
- Visualizes cumulative cost (y) versus time (x).
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# -----------------------------
# 1. PARAMETERS
# -----------------------------
# Epidemic parameters (example values)
M_mild = 160.8                     # daily medical cost per infected (mild)
daily_income = np.array([27.74, 105.12, 92.45])  # children, adults, seniors
employment_rates = np.array([0.0, 0.694, 0.513])  # assume children not employed
vaccination_cost_per_capita = 35.76
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
V3 = (0.01067 - r[0]*V1 - r[1]*V2) / r[2]  # solve for seniors
print("Tentative vaccination rates:", V1, V2, V3)

# For simplicity, we use these constant rates for all times.
V_const = np.array([V1, V2, V3])

# Other epidemiological parameters
gamma = np.array([0.125, 0.083, 0.048])  # recovery rates for three groups
W = 0.005                              # waning immunity rate
# Assume no natural deaths, births, or maturation for simplicity

# Transmission matrix (example fitted matrix)
beta = np.array([[0.1901, 0.0660, 0.0028],
                 [0.1084, 0.1550, 0.0453],
                 [0.1356, 0.1540, 0.1165]])

# -----------------------------
# 2. TIME SETTINGS
# -----------------------------
# Define time periods: five periods over 150 days (t0=0, t1=30, t2=60, t3=90, t4=120, t5=150)
t_periods = [0, 30, 60, 90, 120, 150]
dt = 1.0  # time step in days

# -----------------------------
# 3. MODEL DEFINITION (SIRV with Vaccination)
# -----------------------------
def deriv(y, t, N, beta, gamma, W, V_const):
    """
    Extended SIRV model with constant vaccination rates V_const = [V1, V2, V3].
    Population groups: Children, Adults, Seniors.
    y: vector [S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3]
    """
    S1, I1, R1, V1_state, S2, I2, R2, V2_state, S3, I3, R3, V3_state = y
    # Forcing terms: fraction infected in each group
    lambda1 = beta[0, 0]*I1/N[0] + beta[0, 1]*I2/N[1] + beta[0, 2]*I3/N[2]
    lambda2 = beta[1, 0]*I1/N[0] + beta[1, 1]*I2/N[1] + beta[1, 2]*I3/N[2]
    lambda3 = beta[2, 0]*I1/N[0] + beta[2, 1]*I2/N[1] + beta[2, 2]*I3/N[2]
    
    # Vaccination: constant rate V_const for each group applied to susceptibles
    # Note: V_const is a parameter (not the state V1_state)
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
    # Unpack state for each group:
    S1, I1, R1, V1_state, S2, I2, R2, V2_state, S3, I3, R3, V3_state = y
    # Total infections in each group
    I_total = I1 + I2 + I3
    # Medical cost rate: assume cost is proportional to current infected population
    med_rate = M_mild * I_total
    
    # Wage loss: sum for adults and seniors plus caregiving cost for children.
    # Here we approximate wage loss rate as:
    wage_rate = daily_income[1]*employment_rates[1]*I2 + daily_income[2]*employment_rates[2]*I3
    # For children caregiving, we use adult values:
    wage_rate += daily_income[1]*employment_rates[1]*I1
    
    # Vaccination cost rate: cost incurred by vaccinating susceptibles.
    # Compute daily vaccinations: V_const * S for each group.
    vaccinations = V_const[0]*S1 + V_const[1]*S2 + V_const[2]*S3
    vacc_rate = vaccination_cost_per_capita * vaccinations
    
    # GDP loss rate: per day, scale annual GDP loss fraction by 1/365.
    GDP_loss_fraction = a_param * np.exp(b_param * beta_m) + c_param
    gdp_rate = GDP_per_capita * abs(GDP_loss_fraction) * (total_population / 365)
    
    # For simplicity, we ignore an instantaneous death cost (which is applied as a terminal cost)
    total_rate = med_rate + wage_rate + vacc_rate + gdp_rate
    return total_rate

# -----------------------------
# 5. SIMULATION WITH COST ACCUMULATION OVER PERIODS
# -----------------------------
def simulate_with_cost(V_const, beta, t_periods, N, gamma, W):
    """
    Runs the SIRV model over the defined periods using constant vaccination rates V_const.
    Returns arrays for time and cumulative cost.
    """
    # Initial conditions: assume from file (for simplicity, use populations as initial susceptibles,
    # and very small initial infections; here, we assume:
    S0 = populations.copy()    # all susceptible
    I0 = np.array([10, 10, 10])  # small initial infections in each group
    R0 = np.zeros(3)
    V0 = np.zeros(3)
    y0 = [S0[0], I0[0], R0[0], V0[0],
          S0[1], I0[1], R0[1], V0[1],
          S0[2], I0[2], R0[2], V0[2]]
    
    cumulative_cost = []
    t_cum = []
    total_cost = 0.0
    
    current_state = y0
    current_time = t_periods[0]
    
    for i in range(len(t_periods)-1):
        t_start = t_periods[i]
        t_end = t_periods[i+1]
        t_span = np.arange(t_start, t_end+dt, dt)
        # For this simulation, assume no additional social distancing,
        # so beta multiplier = 1 (or you can adjust if needed)
        beta_m = 1.0
        # Run simulation for this period
        sol = odeint(deriv, current_state, t_span, args=(N, beta, gamma, W, V_const))
        
        # For each day in this period, compute instantaneous cost and accumulate
        for j, t_val in enumerate(t_span):
            y = sol[j]
            cost_rate = instant_cost(y, beta_m)
            total_cost += cost_rate * dt  # dt=1 day
            t_cum.append(t_val)
            cumulative_cost.append(total_cost)
        
        # Update current state to last value of this period for next period simulation
        current_state = sol[-1]
    
    return np.array(t_cum), np.array(cumulative_cost)

# -----------------------------
# 6. RUN THE SIMULATION AND VISUALIZE
# -----------------------------
# Set the time periods: 0, 30, 60, 90, 120, 150 days
t_cum, cum_cost = simulate_with_cost(V_const, beta, t_periods, populations, gamma, W)

# Plot cumulative cost vs time
plt.figure(figsize=(10, 6))
plt.plot(t_cum, cum_cost, marker='o', linestyle='-', color='purple')
plt.xlabel('Time (days)')
plt.ylabel('Cumulative Economic Cost (USD)')
plt.title('Cumulative Economic Cost vs. Time (150-day Simulation)')
plt.grid(True)
plt.show()

# Print final total cost:
print("Total Economic Cost over 150 days: ${:,.2f}".format(cum_cost[-1]))

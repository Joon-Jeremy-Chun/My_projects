# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 00:30:25 2025

Piecewise Vaccination Rate Scenario Comparison (No Distancing)

Two scenarios are compared (β multiplier = 1.0 for both):
1. Scenario 1 (Uniform): The same vaccination rate vector is used in all periods.
2. Scenario 2 (Piecewise): Different vaccination rate vectors are used in each period.
Each vaccination vector [V1,V2,V3] is chosen so that:
    0.1207*V1 + 0.6092*V2 + 0.27*V3 = 0.01067.
The simulation runs for 150 days divided into five 30-day periods.
The output is a plot of cumulative economic cost vs. time for both scenarios.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# -----------------------------
# 1. PARAMETERS
# -----------------------------
M_mild = 160.8                     # daily medical cost per infected (mild)
daily_income = np.array([27.74, 105.12, 92.45])
employment_rates = np.array([0.0, 0.694, 0.513])
funeral_cost = 9405.38
GDP_per_capita = 31929
a_param, b_param, c_param = -0.3648, -8.7989, -0.0012

populations = np.array([6246073, 31530507, 13972160])
total_population = np.sum(populations)
r = np.array([0.1207, 0.6092, 0.27])
daily_vacc_target = 0.01067
vaccination_cost_per_capita = 35.76

gamma = np.array([0.125, 0.083, 0.048])
W = 0.005

beta = np.array([[0.1901, 0.0660, 0.0028],
                 [0.1084, 0.1550, 0.0453],
                 [0.1356, 0.1540, 0.1165]])
beta_mod = beta * 1.0  # no distancing

# -----------------------------
# 2. TIME SETTINGS
# -----------------------------
t_periods = [0, 30, 60, 90, 120, 150]
dt = 1.0

# -----------------------------
# 3. MODEL DEFINITION (SIRV with Vaccination)
# -----------------------------
def deriv(y, t, N, beta, gamma, W, V_current):
    S1, I1, R1, V1_state, S2, I2, R2, V2_state, S3, I3, R3, V3_state = y
    lambda1 = beta[0, 0]*I1/N[0] + beta[0, 1]*I2/N[1] + beta[0, 2]*I3/N[2]
    lambda2 = beta[1, 0]*I1/N[0] + beta[1, 1]*I2/N[1] + beta[1, 2]*I3/N[2]
    lambda3 = beta[2, 0]*I1/N[0] + beta[2, 1]*I2/N[1] + beta[2, 2]*I3/N[2]
    
    dS1dt = -lambda1 * S1 - V_current[0]*S1 + W*(R1 + V1_state)
    dI1dt = lambda1 * S1 - gamma[0]*I1
    dR1dt = gamma[0]*I1 - W*R1
    dV1dt = V_current[0]*S1 - W*V1_state

    dS2dt = -lambda2 * S2 - V_current[1]*S2 + W*(R2 + V2_state)
    dI2dt = lambda2 * S2 - gamma[1]*I2
    dR2dt = gamma[1]*I2 - W*R2
    dV2dt = V_current[1]*S2 - W*V2_state

    dS3dt = -lambda3 * S3 - V_current[2]*S3 + W*(R3 + V3_state)
    dI3dt = lambda3 * S3 - gamma[2]*I3
    dR3dt = gamma[2]*I3 - W*R3
    dV3dt = V_current[2]*S3 - W*V3_state

    return [dS1dt, dI1dt, dR1dt, dV1dt,
            dS2dt, dI2dt, dR2dt, dV2dt,
            dS3dt, dI3dt, dR3dt, dV3dt]

# -----------------------------
# 4. COST FUNCTION (Instantaneous Cost Rate)
# -----------------------------
def instant_cost(y, beta_m, V_current):
    S1, I1, R1, V1_state, S2, I2, R2, V2_state, S3, I3, R3, V3_state = y
    I_total = I1 + I2 + I3
    med_rate = M_mild * I_total
    wage_rate = daily_income[1]*employment_rates[1]*(I2 + I1) + daily_income[2]*employment_rates[2]*I3
    vaccinations = V_current[0]*S1 + V_current[1]*S2 + V_current[2]*S3
    vacc_rate = vaccination_cost_per_capita * vaccinations
    GDP_loss_fraction = a_param * np.exp(b_param * beta_m) + c_param
    gdp_rate = GDP_per_capita * abs(GDP_loss_fraction) * (total_population / 365)
    total_rate = med_rate + wage_rate + vacc_rate + gdp_rate
    return total_rate

# -----------------------------
# 5. SIMULATION FUNCTION WITH PIECEWISE VACCINATION
# -----------------------------
def simulate_piecewise(V_schedule, beta_mod, t_periods, populations, gamma, W):
    S0 = populations.copy()
    I0 = np.array([10, 10, 10])
    R0 = np.zeros(3)
    V0 = np.zeros(3)
    y0 = [S0[0], I0[0], R0[0], V0[0],
          S0[1], I0[1], R0[1], V0[1],
          S0[2], I0[2], R0[2], V0[2]]
    
    cumulative_cost = []
    t_cum = []
    total_cost = 0.0
    current_state = y0
    beta_multiplier = 1.0  # no distancing
    
    for i in range(len(t_periods)-1):
        t_start = t_periods[i]
        t_end = t_periods[i+1]
        t_span = np.arange(t_start, t_end+dt, dt)
        V_current = V_schedule[i]
        sol = odeint(deriv, current_state, t_span, args=(populations, beta, gamma, W, V_current))
        
        for t_val, y in zip(t_span, sol):
            cost_rate = instant_cost(y, beta_multiplier, V_current)
            total_cost += cost_rate * dt
            t_cum.append(t_val)
            cumulative_cost.append(total_cost)
        
        current_state = sol[-1]
    
    return np.array(t_cum), np.array(cumulative_cost)

# -----------------------------
# 6. DEFINE VACCINATION SCHEDULES FOR TWO SCENARIOS
# -----------------------------
def compute_V3(V1, V2):
    return (daily_vacc_target - r[0]*V1 - r[1]*V2) / r[2]

# Scenario 1: Uniform schedule (same vaccination vector for all periods)
V1_s1 = 0.001
V2_s1 = 0.005
V3_s1 = compute_V3(V1_s1, V2_s1)
V_schedule_s1 = [np.array([V1_s1, V2_s1, V3_s1]) for _ in range(len(t_periods)-1)]
print("Scenario 1 vaccination rates (each period):", V_schedule_s1[0])

# Scenario 2: Piecewise schedule (different rates per period)
V_schedule_s2 = []
# Period 1 (0-30): Prioritize adults (higher V2)
V1_s2_1 = 0.0005; V2_s2_1 = 0.007; V3_s2_1 = compute_V3(V1_s2_1, V2_s2_1)
V_schedule_s2.append(np.array([V1_s2_1, V2_s2_1, V3_s2_1]))
# Period 2 (30-60): More balanced
V1_s2_2 = 0.0010; V2_s2_2 = 0.0050; V3_s2_2 = compute_V3(V1_s2_2, V2_s2_2)
V_schedule_s2.append(np.array([V1_s2_2, V2_s2_2, V3_s2_2]))
# Period 3 (60-90): Increase for children
V1_s2_3 = 0.0015; V2_s2_3 = 0.0050; V3_s2_3 = compute_V3(V1_s2_3, V2_s2_3)
V_schedule_s2.append(np.array([V1_s2_3, V2_s2_3, V3_s2_3]))
# Period 4 (90-120): Favor adults moderately
V1_s2_4 = 0.0010; V2_s2_4 = 0.0060; V3_s2_4 = compute_V3(V1_s2_4, V2_s2_4)
V_schedule_s2.append(np.array([V1_s2_4, V2_s2_4, V3_s2_4]))
# Period 5 (120-150): Return to uniform
V1_s2_5 = 0.0010; V2_s2_5 = 0.0050; V3_s2_5 = compute_V3(V1_s2_5, V2_s2_5)
V_schedule_s2.append(np.array([V1_s2_5, V2_s2_5, V3_s2_5]))

print("Scenario 2 vaccination rates, period 1:", V_schedule_s2[0])
print("Scenario 2 vaccination rates, period 2:", V_schedule_s2[1])
print("Scenario 2 vaccination rates, period 3:", V_schedule_s2[2])
print("Scenario 2 vaccination rates, period 4:", V_schedule_s2[3])
print("Scenario 2 vaccination rates, period 5:", V_schedule_s2[4])

# -----------------------------
# 7. RUN SIMULATIONS FOR BOTH SCENARIOS
# -----------------------------
t_cum_s1, cum_cost_s1 = simulate_piecewise(V_schedule_s1, beta_mod, t_periods, populations, gamma, W)
t_cum_s2, cum_cost_s2 = simulate_piecewise(V_schedule_s2, beta_mod, t_periods, populations, gamma, W)

# -----------------------------
# 8. VISUALIZATION: CUMULATIVE COST VS TIME
# -----------------------------
plt.figure(figsize=(10, 6))
plt.plot(t_cum_s1, cum_cost_s1, label='Scenario 1 (Uniform Vaccination)', linestyle='-')
plt.plot(t_cum_s2, cum_cost_s2, label='Scenario 2 (Piecewise Vaccination)', linestyle='--')
plt.xlabel('Time (days)')
plt.ylabel('Cumulative Economic Cost (USD)')
plt.title('Cumulative Economic Cost vs. Time (150-day Simulation, No Distancing)')
plt.legend()
plt.grid(True)
plt.show()

print("Total Economic Cost over 150 days (Scenario 1): ${:,.2f}".format(cum_cost_s1[-1]))
print("Total Economic Cost over 150 days (Scenario 2): ${:,.2f}".format(cum_cost_s2[-1]))

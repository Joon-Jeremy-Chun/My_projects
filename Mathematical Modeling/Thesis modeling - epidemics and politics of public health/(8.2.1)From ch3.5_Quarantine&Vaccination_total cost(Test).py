# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 14:44:51 2025

@author: joonc
"""

# -*- coding: utf-8 -*-
"""
Created on [Date]

Author: joonc

Integrated Model with Intervention Metrics:
  - Uses transmission and dynamic vaccination strategy parameters (from the second code)
  - Incorporates economic cost calculations (from the first code)
  - Computes peak infections and peak day for each scenario
  - Scenarios:
      1. Social Distancing Only (dynamic vaccination)
      2. Vaccination Only (manual constant vaccination)
      3. Vaccination with Light Social Distancing (manual vaccination)
  
Outcomes: Infection and cost curves (with peak markers) plus printed summary metrics.
"""

import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ---------------------------
# Economic Cost Parameters
# ---------------------------
m, h, i = 0.8, 0.15, 0.05          # Ratios for medical cases (mild, hospitalized, intensive)
C_M, C_H, C_I = 5, 432.5, 1129.5     # Costs for medical treatment (in $)
VC, PC, LC = 20, 10, 5              # Vaccination costs (in $)
E = [0.04, 0.69, 0.051]             # Employment rates for each group
D_in = [27.73, 105.12, 92.45]       # Average daily income for each group (in $)
FC = 9405.38                      # Funeral cost (in $)

# ---------------------------
# 1. LOAD EPIDEMIC PARAMETERS
# ---------------------------
def load_parameters(file_path):
    df = pd.read_excel(file_path, header=None)
    # Recovery rates (for Children, Adults, Seniors)
    gamma = df.iloc[4, 0:3].values
    # Here "mu" is used as a secondary rate (e.g., maturation or transition); same as in your second code
    mu = df.iloc[6, 0:2].values  
    # Waning immunity rate
    W = df.iloc[8, 0]
    # Time span (in days)
    time_span = df.iloc[12, 0]
    # Population sizes and initial conditions for three groups
    N = df.iloc[14, 0:3].values
    S_init = df.iloc[14, 0:3].values
    I_init = df.iloc[16, 0:3].values  
    R_init = df.iloc[18, 0:3].values   
    V_init = df.iloc[20, 0:3].values
    return gamma, mu, W, time_span, N, S_init, I_init, R_init, V_init

file_path = 'Inputs.xlsx'
gamma, mu, W, time_span, N, S_init, I_init, R_init, V_init = load_parameters(file_path)

# Load transmission rates matrix from CSV (as in second code)
transmission_rates_csv_path = 'DataSets/Fitted_Beta_Matrix.csv'
beta = pd.read_csv(transmission_rates_csv_path).iloc[0:3, 1:4].values

# Initial conditions for 3 groups (order: S, I, R, V for each group)
initial_conditions = [
    S_init[0], I_init[0], R_init[0], V_init[0],
    S_init[1], I_init[1], R_init[1], V_init[1],
    S_init[2], I_init[2], R_init[2], V_init[2]
]

# ---------------------------
# 2. VACCINATION STRATEGY SETUP
# ---------------------------
# Manual vaccination rate (for scenarios where use_dynamic=False)
vaccination_rates = np.array([0.0, 0.0, 0.0])  # default; will be overridden in Scenario 2 & 3

# For dynamic vaccination, you might set alternative values; here we simply use the same rates.
vaccination_rates_dynamic = vaccination_rates

# Parameters for dynamic vaccination (phases)
t1, t2 = 30, 60
B_timeP0, B_timeP1, B_timeP2 = 1.0, 1.0, 1.0

def vaccination_strategy(t, use_dynamic=True):
    """Return the vaccination rate vector at time t."""
    rates = vaccination_rates_dynamic if use_dynamic else vaccination_rates
    if t < t1:
        return rates * B_timeP0
    elif t1 <= t < t2:
        return rates * B_timeP1
    else:
        return rates * B_timeP2

# ---------------------------
# 3. MODEL DEFINITION (SIRV)
# ---------------------------
def deriv(y, t, N, beta, gamma, mu, W, use_dynamic):
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y
    a_t = vaccination_strategy(t, use_dynamic)
    
    # Force of infection for each group
    lambda1 = beta[0,0]*I1/N[0] + beta[0,1]*I2/N[1] + beta[0,2]*I3/N[2]
    lambda2 = beta[1,0]*I1/N[0] + beta[1,1]*I2/N[1] + beta[1,2]*I3/N[2]
    lambda3 = beta[2,0]*I1/N[0] + beta[2,1]*I2/N[1] + beta[2,2]*I3/N[2]
    
    dS1dt = -lambda1*S1 - a_t[0]*S1 + W*R1 + W*V1
    dI1dt = lambda1*S1 - gamma[0]*I1 - mu[0]*I1
    dR1dt = gamma[0]*I1 - W*R1 - mu[0]*R1
    dV1dt = a_t[0]*S1 - W*V1
    
    dS2dt = -lambda2*S2 + mu[0]*S1 - a_t[1]*S2 + W*R2 + W*V2
    dI2dt = lambda2*S2 + mu[0]*I1 - gamma[1]*I2 - mu[1]*I2
    dR2dt = gamma[1]*I2 + mu[0]*R1 - W*R2 - mu[1]*R2
    dV2dt = a_t[1]*S2 - W*V2
    
    dS3dt = -lambda3*S3 + mu[1]*S2 - a_t[2]*S3 + W*R3 + W*V3
    dI3dt = lambda3*S3 + mu[1]*I2 - gamma[2]*I3
    dR3dt = gamma[2]*I3 + mu[1]*R2 - W*R3
    dV3dt = a_t[2]*S3 - W*V3
    
    return [dS1dt, dI1dt, dR1dt, dV1dt,
            dS2dt, dI2dt, dR2dt, dV2dt,
            dS3dt, I3dt, R3dt, V3dt] if (lambda3:=beta[2,0]*I1/N[0] + beta[2,1]*I2/N[1] + beta[2,2]*I3/N[2]) is not None else None

# Since we want a clean code, I'll reassign variables below:
def deriv(y, t, N, beta, gamma, mu, W, use_dynamic):
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y
    a_t = vaccination_strategy(t, use_dynamic)
    
    # Force of infection for each group
    lambda1 = beta[0,0]*I1/N[0] + beta[0,1]*I2/N[1] + beta[0,2]*I3/N[2]
    lambda2 = beta[1,0]*I1/N[0] + beta[1,1]*I2/N[1] + beta[1,2]*I3/N[2]
    lambda3 = beta[2,0]*I1/N[0] + beta[2,1]*I2/N[1] + beta[2,2]*I3/N[2]
    
    dS1dt = -lambda1*S1 - a_t[0]*S1 + W*R1 + W*V1
    dI1dt = lambda1*S1 - gamma[0]*I1 - mu[0]*I1
    dR1dt = gamma[0]*I1 - W*R1 - mu[0]*R1
    dV1dt = a_t[0]*S1 - W*V1
    
    dS2dt = -lambda2*S2 + mu[0]*S1 - a_t[1]*S2 + W*R2 + W*V2
    dI2dt = lambda2*S2 + mu[0]*I1 - gamma[1]*I2 - mu[1]*I2
    dR2dt = gamma[1]*I2 + mu[0]*R1 - W*R2 - mu[1]*R2
    dV2dt = a_t[1]*S2 - W*V2
    
    dS3dt = -lambda3*S3 + mu[1]*S2 - a_t[2]*S3 + W*R3 + W*V3
    dI3dt = lambda3*S3 + mu[1]*I2 - gamma[2]*I3
    dR3dt = gamma[2]*I3 + mu[1]*R2 - W*R3
    dV3dt = a_t[2]*S3 - W*V3
    
    return [dS1dt, dI1dt, dR1dt, dV1dt,
            dS2dt, dI2dt, dR2dt, dV2dt,
            dS3dt, dI3dt, dR3dt, dV3dt]

# ---------------------------
# 4. ECONOMIC COST FUNCTION
# ---------------------------
def calculate_costs(I, V, D=0):
    """Compute the total economic cost from group infections and vaccinations."""
    medical_cost = (I[0] + I[1] + I[2]) * (m * C_M + h * C_H + i * C_I)
    vaccination_cost = (V[0] + V[1] + V[2]) * (VC + PC + LC)
    productivity_cost = sum(I[j] * E[j] * D_in[j] for j in range(3))
    productivity_cost += I[0] * E[1] * D_in[1]  # extra loss for Group 2 due to Group 1 infections
    funeral_cost = D * FC if D > 0 else 0
    total_cost = medical_cost + vaccination_cost + productivity_cost + funeral_cost
    return total_cost

def compute_economic_costs(sol, t_array):
    """Given simulation results and time vector, compute economic cost at each time."""
    costs = []
    for idx, _ in enumerate(t_array):
        I_vals = [sol[idx, 1], sol[idx, 5], sol[idx, 9]]
        V_vals = [sol[idx, 3], sol[idx, 7], sol[idx, 11]]
        cost = calculate_costs(I_vals, V_vals, D=0)
        costs.append(cost)
    return np.array(costs)

# ---------------------------
# 5. SIMULATION FUNCTION
# ---------------------------
def run_simulation(beta_mod, use_dynamic):
    t = np.linspace(0, time_span, int(time_span))
    sol = odeint(deriv, initial_conditions, t, args=(N, beta_mod, gamma, mu, W, use_dynamic))
    econ_costs = compute_economic_costs(sol, t)
    I_total = sol[:, 1] + sol[:, 5] + sol[:, 9]
    peak_infected = np.max(I_total)
    peak_day = t[np.argmax(I_total)]
    return t, sol, econ_costs, peak_infected, peak_day

# ---------------------------
# 6. SCENARIO SIMULATIONS
# ---------------------------

# Scenario 1: Social Distancing Only (dynamic vaccination)
beta_multiplier = 0.9  # Uniform reduction
beta_mod_1 = beta * beta_multiplier
t1_sim, sol1, econ_costs1, peak_inf1, peak_day1 = run_simulation(beta_mod_1, use_dynamic=True)
I_total1 = sol1[:,1] + sol1[:,5] + sol1[:,9]
final_cost1 = econ_costs1[-1]

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(t1_sim, I_total1, label='Total Infected')
peak_idx1 = np.argmax(I_total1)
plt.scatter(t1_sim[peak_idx1], I_total1[peak_idx1], color='red',
            label=f'Peak: {I_total1[peak_idx1]:.0f} on day {t1_sim[peak_idx1]:.0f}')
plt.xlabel('Days')
plt.ylabel('Infected Individuals')
plt.title('Scenario 1: Social Distancing Only')
plt.legend()

plt.subplot(1,2,2)
plt.plot(t1_sim, econ_costs1, label='Economic Cost', color='red')
plt.xlabel('Days')
plt.ylabel('Total Economic Cost ($)')
plt.title('Scenario 1: Economic Cost Over Time')
plt.legend()

plt.tight_layout()
plt.show()
print("Scenario 1:")
print("  Peak Infected: {:.0f} on day {:.0f}".format(peak_inf1, peak_day1))
print("  Final Economic Cost: ${:,.2f}\n".format(final_cost1))

# Scenario 2: Vaccination Only (manual constant vaccination)
vaccination_rates = np.array([0.01, 0.01, 0.01])  # constant vaccination rate
beta_mod_2 = beta  # No modification
t2_sim, sol2, econ_costs2, peak_inf2, peak_day2 = run_simulation(beta_mod_2, use_dynamic=False)
I_total2 = sol2[:,1] + sol2[:,5] + sol2[:,9]
final_cost2 = econ_costs2[-1]

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(t2_sim, I_total2, label='Total Infected')
peak_idx2 = np.argmax(I_total2)
plt.scatter(t2_sim[peak_idx2], I_total2[peak_idx2], color='red',
            label=f'Peak: {I_total2[peak_idx2]:.0f} on day {t2_sim[peak_idx2]:.0f}')
plt.xlabel('Days')
plt.ylabel('Infected Individuals')
plt.title('Scenario 2: Vaccination Only')
plt.legend()

plt.subplot(1,2,2)
plt.plot(t2_sim, econ_costs2, label='Economic Cost', color='red')
plt.xlabel('Days')
plt.ylabel('Total Economic Cost ($)')
plt.title('Scenario 2: Economic Cost Over Time')
plt.legend()

plt.tight_layout()
plt.show()
print("Scenario 2:")
print("  Peak Infected: {:.0f} on day {:.0f}".format(peak_inf2, peak_day2))
print("  Final Economic Cost: ${:,.2f}\n".format(final_cost2))

# Scenario 3: Vaccination with Light Social Distancing (manual vaccination)
beta_multiplier_combined = 0.9
beta_mod_3 = beta * beta_multiplier_combined
t3_sim, sol3, econ_costs3, peak_inf3, peak_day3 = run_simulation(beta_mod_3, use_dynamic=False)
I_total3 = sol3[:,1] + sol3[:,5] + sol3[:,9]
final_cost3 = econ_costs3[-1]

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(t3_sim, I_total3, label='Total Infected')
peak_idx3 = np.argmax(I_total3)
plt.scatter(t3_sim[peak_idx3], I_total3[peak_idx3], color='red',
            label=f'Peak: {I_total3[peak_idx3]:.0f} on day {t3_sim[peak_idx3]:.0f}')
plt.xlabel('Days')
plt.ylabel('Infected Individuals')
plt.title('Scenario 3: Vaccination with Light Social Distancing')
plt.legend()

plt.subplot(1,2,2)
plt.plot(t3_sim, econ_costs3, label='Economic Cost', color='red')
plt.xlabel('Days')
plt.ylabel('Total Economic Cost ($)')
plt.title('Scenario 3: Economic Cost Over Time')
plt.legend()

plt.tight_layout()
plt.show()
print("Scenario 3:")
print("  Peak Infected: {:.0f} on day {:.0f}".format(peak_inf3, peak_day3))
print("  Final Economic Cost: ${:,.2f}".format(final_cost3))

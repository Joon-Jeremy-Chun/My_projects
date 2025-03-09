# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 14:15:41 2025

@author: joonc
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:34:00 2024

Author: joonc
"""
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# Function to load model parameters and initial conditions from Excel
def load_parameters(file_path):
    df = pd.read_excel(file_path, header=None)
    
    # Transmission rates matrix (3x3)
    transmission_rates = df.iloc[0:3, 0:3].values
    
    # Recovery rates (3 values)
    recovery_rates = df.iloc[4, 0:3].values
    
    # Maturity rates (2 values)
    maturity_rates = df.iloc[6, 0:2].values
    
    # Waning immunity rate (single value)
    waning_immunity_rate = df.iloc[8, 0]
    
    # Vaccination rates (3 values)
    vaccination_rates = df.iloc[10, 0:3].values
    
    # Time span (in days)
    time_span = df.iloc[12, 0]
    
    # Population sizes (3 values)
    population_size = df.iloc[14, 0:3].values  
    
    # Initial conditions: Susceptible, Infectious, Recovered, Vaccinated (each for 3 groups)
    susceptible_init = df.iloc[14, 0:3].values  # Define population size
    infectious_init = df.iloc[16, 0:3].values  
    recovered_init = df.iloc[18, 0:3].values   
    vaccinated_init = df.iloc[20, 0:3].values  
    
    # Quarantine-related data
    under_quarantine_transmission_rates = df.iloc[22:26, 0:3].values
    quarantine_day = df.iloc[26, 0]
    
    return (transmission_rates, recovery_rates, maturity_rates, waning_immunity_rate,
            vaccination_rates, time_span, population_size, susceptible_init, infectious_init,
            recovered_init, vaccinated_init, under_quarantine_transmission_rates, quarantine_day)

#%% Load the parameters from the updated Excel file
file_path = 'Inputs.xlsx'  # Ensure the correct path to the Excel file
(beta, gamma, mu, W, a, time_span, N, S_init, I_init, R_init, V_init, beta_quarantine, quarantine_day) = load_parameters(file_path)

# Economic parameters
m, h, i = 0.8, 0.15, 0.05  # Ratios for medical cases (mild, hospitalized, intensive)
C_M, C_H, C_I = 5, 432.5, 1129.5  # Costs for medical treatment (in $)
VC, PC, LC = 20, 10, 5  # Vaccination costs (in $)
E = [0.04, 0.69, 0.051]  # Employment rates for each group
D_in = [27.73, 105.12, 92.45]  # Average daily income for each group (in $)
FC = 9405.38  # Funeral cost (in $)

# Group indices for readability (not strictly used later)
S1, I1, R1, V1 = 0, 1, 2, 3  # Group 1
S2, I2, R2, V2 = 4, 5, 6, 7  # Group 2
S3, I3, R3, V3 = 8, 9, 10, 11  # Group 3

# Initial conditions for the three groups using values from the Excel file
initial_conditions = [
    float(S_init[0]), float(I_init[0]), float(R_init[0]), float(V_init[0]),  # Group 1: S, I, R, V
    float(S_init[1]), float(I_init[1]), float(R_init[1]), float(V_init[1]),  # Group 2: S, I, R, V
    float(S_init[2]), float(I_init[2]), float(R_init[2]), float(V_init[2])   # Group 3: S, I, R, V
]

#%% SIRV model differential equations including quarantine dynamics and economic costs
def deriv(y, t, N, beta, beta_quarantine, quarantine_day, gamma, mu, W, a):
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = [float(val) for val in y]
    
    try:
        # Apply quarantine transmission rates after a specific day
        if t >= quarantine_day:
            current_beta = beta_quarantine
        else:
            current_beta = beta
        
        # Force of infection (λ_i) for each group
        lambda1 = current_beta[0, 0] * I1/float(N[0]) + current_beta[0, 1] * I2/float(N[1]) + current_beta[0, 2] * I3/float(N[2])
        lambda2 = current_beta[1, 0] * I1/float(N[0]) + current_beta[1, 1] * I2/float(N[1]) + current_beta[1, 2] * I3/float(N[2])
        lambda3 = current_beta[2, 0] * I1/float(N[0]) + current_beta[2, 1] * I2/float(N[1]) + current_beta[2, 2] * I3/float(N[2])
        
        # Differential equations for each compartment
        # Group 1 (e.g., children)
        dS1dt = -lambda1 * S1 - a[0] * S1 + W * R1 + W * V1
        dI1dt = lambda1 * S1 - gamma[0] * I1 - mu[0] * I1
        dR1dt = gamma[0] * I1 - W * R1 - mu[0] * R1
        dV1dt = a[0] * S1 - W * V1
        
        # Group 2 (e.g., adults)
        dS2dt = -lambda2 * S2 + mu[0] * S1 - a[1] * S2 + W * R2 + W * V2
        dI2dt = lambda2 * S2 + mu[0] * I1 - gamma[1] * I2 - mu[1] * I2
        dR2dt = gamma[1] * I2 + mu[0] * R1 - W * R2 - mu[1] * R2
        dV2dt = a[1] * S2 - W * V2
        
        # Group 3 (e.g., seniors)
        dS3dt = -lambda3 * S3 + mu[1] * S2 - a[2] * S3 + W * R3 + W * V3
        dI3dt = lambda3 * S3 + mu[1] * I2 - gamma[2] * I3
        dR3dt = gamma[2] * I3 + mu[1] * R2 - W * R3
        dV3dt = a[2] * S3 - W * V3
        
        return dS1dt, dI1dt, dR1dt, dV1dt, dS2dt, dI2dt, dR2dt, dV2dt, dS3dt, dI3dt, dR3dt, dV3dt
    
    except Exception as e:
        print(f"Error in deriv function at time t = {t}")
        print(f"Current state y: {y}")
        print(f"Parameters: N = {N}, beta = {beta}, beta_quarantine = {beta_quarantine}, gamma = {gamma}, mu = {mu}, W = {W}, a = {a}")
        raise e

#%% Economic cost calculations (total cost)
def calculate_costs(I, V, D, t):
    # Medical cost: aggregated over groups
    medical_cost = (I[0] + I[1] + I[2]) * (m * C_M + h * C_H + i * C_I)
    
    # Vaccination cost: aggregated over groups
    vaccination_cost = (V[0] + V[1] + V[2]) * (VC + PC + LC)
    
    # Productivity loss cost: includes extra loss for Group 2 due to Group 1 infections
    productivity_cost = sum(I[j] * E[j] * D_in[j] for j in range(3))
    productivity_cost += I[0] * E[1] * D_in[1]  # Additional loss for Group 2
    
    # Funeral cost (if applicable)
    funeral_cost = D * FC if D > 0 else 0
    
    # Total cost
    total_cost = medical_cost + vaccination_cost + productivity_cost + funeral_cost
    
    return total_cost

# New function: Economic cost calculations by group
def calculate_costs_by_group(I, V, D, t):
    # Medical cost per group
    medical_cost_1 = I[0] * (m * C_M + h * C_H + i * C_I)
    medical_cost_2 = I[1] * (m * C_M + h * C_H + i * C_I)
    medical_cost_3 = I[2] * (m * C_M + h * C_H + i * C_I)
    
    # Vaccination cost per group
    vaccination_cost_1 = V[0] * (VC + PC + LC)
    vaccination_cost_2 = V[1] * (VC + PC + LC)
    vaccination_cost_3 = V[2] * (VC + PC + LC)
    
    # Productivity loss cost per group
    prod_cost_1 = I[0] * E[0] * D_in[0]
    # Group 2: own loss plus additional loss due to Group 1 infections
    prod_cost_2 = I[1] * E[1] * D_in[1] + I[0] * E[1] * D_in[1]
    prod_cost_3 = I[2] * E[2] * D_in[2]
    
    # Total cost for each group
    cost_group1 = medical_cost_1 + vaccination_cost_1 + prod_cost_1
    cost_group2 = medical_cost_2 + vaccination_cost_2 + prod_cost_2
    cost_group3 = medical_cost_3 + vaccination_cost_3 + prod_cost_3
    
    # Funeral cost applied if applicable (added to total cost)
    funeral_cost = D * FC if D > 0 else 0
    
    total_cost = cost_group1 + cost_group2 + cost_group3 + funeral_cost
    return cost_group1, cost_group2, cost_group3, total_cost

#%% Time grid (in days)
t = np.linspace(0, time_span, int(time_span))

# Integrate the SIRV equations over time with quarantine logic
results = odeint(deriv, initial_conditions, t, args=(N, beta, beta_quarantine, quarantine_day, gamma, mu, W, a))

# Extract results for each compartment
S1_sol, I1_sol, R1_sol, V1_sol, S2_sol, I2_sol, R2_sol, V2_sol, S3_sol, I3_sol, R3_sol, V3_sol = results.T

# Identify local maxima for the infected groups
def find_local_maxima(I):
    maxima_indices = argrelextrema(I, np.greater)[0]
    maxima_values = I[maxima_indices]
    return maxima_indices, maxima_values

maxima_I1_idx, maxima_I1_val = find_local_maxima(I1_sol)
maxima_I2_idx, maxima_I2_val = find_local_maxima(I2_sol)
maxima_I3_idx, maxima_I3_val = find_local_maxima(I3_sol)

#%% Calculate total economic costs over time using the original function
total_costs = []
for idx in range(len(t)):
    I_current = [I1_sol[idx], I2_sol[idx], I3_sol[idx]]
    V_current = [V1_sol[idx], V2_sol[idx], V3_sol[idx]]
    D = 0  # Assuming no death component in current implementation
    cost = calculate_costs(I_current, V_current, D, t[idx])
    total_costs.append(cost)

#%% Calculate economic cost breakdown by group over time using the new function
costs_group1 = []
costs_group2 = []
costs_group3 = []
total_costs_by_group = []
for idx in range(len(t)):
    I_current = [I1_sol[idx], I2_sol[idx], I3_sol[idx]]
    V_current = [V1_sol[idx], V2_sol[idx], V3_sol[idx]]
    D = 0  # Assuming no death component in current implementation
    cg1, cg2, cg3, tot = calculate_costs_by_group(I_current, V_current, D, t[idx])
    costs_group1.append(cg1)
    costs_group2.append(cg2)
    costs_group3.append(cg3)
    total_costs_by_group.append(tot)

#%% Plot SIRV dynamics for each group with local maxima annotations
plt.figure(figsize=(14, 16))
for i, (S, I, R, V, maxima_idx, maxima_val, group) in enumerate(zip(
        [S1_sol, S2_sol, S3_sol], [I1_sol, I2_sol, I3_sol], [R1_sol, R2_sol, R3_sol], [V1_sol, V2_sol, V3_sol],
        [maxima_I1_idx, maxima_I2_idx, maxima_I3_idx],
        [maxima_I1_val, maxima_I2_val, maxima_I3_val],
        ['Group 1', 'Group 2', 'Group 3']), start=1):
    
    plt.subplot(4, 1, i)
    plt.plot(t, S, 'b', label=f'Susceptible ({group})')
    plt.plot(t, I, 'r', label=f'Infected ({group})')
    plt.plot(t, R, 'g', label=f'Recovered ({group})')
    plt.plot(t, V, 'm', label=f'Vaccinated ({group})')

    # Highlight local maxima for infected group
    plt.scatter(t[maxima_idx], maxima_val, color='black', zorder=5, label=f'Local Maxima ({group} Infected)')
    
    # Annotate maxima with coordinates
    for idx_val, val in zip(maxima_idx, maxima_val):
        plt.annotate(f'({t[idx_val]:.1f}, {val:.1f})', (t[idx_val], val), textcoords="offset points", xytext=(0, 10), ha='center')
    
    plt.title(f'SIRV Dynamics for {group} with Quarantine Applied on Day {quarantine_day}')
    plt.xlabel('Days')
    plt.ylabel('Population')
    plt.legend()

# Plot total economic costs over time (from original function)
plt.subplot(4, 1, 4)
plt.plot(t, total_costs, 'k', label='Total Economic Cost')
plt.title('Total Economic Cost Over Time')
plt.xlabel('Days')
plt.ylabel('Cost ($)')
plt.legend()

plt.tight_layout()
plt.show()

#%% Plot economic cost breakdown by group over time
plt.figure(figsize=(10, 6))
plt.plot(t, costs_group1, label='Group 1 Cost')
plt.plot(t, costs_group2, label='Group 2 Cost')
plt.plot(t, costs_group3, label='Group 3 Cost')
plt.plot(t, total_costs_by_group, 'k--', linewidth=2, label='Total Economic Cost')
plt.xlabel('Days')
plt.ylabel('Cost ($)')
plt.title('Economic Cost Breakdown by Group Over Time')
plt.legend()
plt.tight_layout()
plt.show()

#%% Additional Code: Display final economic cost breakdown for the three groups and total cost
final_group1_cost = costs_group1[-1]
final_group2_cost = costs_group2[-1]
final_group3_cost = costs_group3[-1]
final_total_cost = total_costs_by_group[-1]

print("Final Economic Cost Breakdown at t = {:.2f} days:".format(t[-1]))
print("Group 1 Cost: ${:,.2f}".format(final_group1_cost))
print("Group 2 Cost: ${:,.2f}".format(final_group2_cost))
print("Group 3 Cost: ${:,.2f}".format(final_group3_cost))
print("Total Economic Cost: ${:,.2f}".format(final_total_cost))

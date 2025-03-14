# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:34:00 2024

@author: joonc
"""
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# Function to load model parameters and initial conditions from Excel
def load_parameters(file_path):
    df = pd.read_excel(file_path, header=None)
    
    # Transmission rates matrix (3x3) from Excel (not used for beta)
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

#%%
# Load the parameters from the updated Excel file
file_path = 'Inputs.xlsx'  # Ensure the correct path to the Excel file
# The beta matrix from Excel is ignored because beta is always loaded from CSV.
(_, gamma, mu, W, a, time_span, N, S_init, I_init, R_init, V_init, beta_quarantine, quarantine_day) = load_parameters(file_path)

# Now always load beta from the CSV file
beta_csv_path = 'DataSets/Fitted_Beta_Matrix.csv'
beta = pd.read_csv(beta_csv_path).iloc[0:3, 1:4].values

# Economic parameters
m, h, i = 0.8, 0.15, 0.05  # Ratios for medical cases (mild, hospitalized, intensive)
C_M, C_H, C_I = 5, 432.5, 1129.5  # Costs for medical treatment (in $)
VC, PC, LC = 20, 10, 5  # Vaccination costs (in $)
E = [0.04, 0.69, 0.051]  # Employment rates for each group
D_in = [27.73, 105.12, 92.45]  # Average daily income for each group (in $)
FC = 9405.38  # Funeral cost (in $)

# Group indices for readability
S1, I1, R1, V1 = 0, 1, 2, 3  # First group (e.g., children)
S2, I2, R2, V2 = 4, 5, 6, 7  # Second group (e.g., adults)
S3, I3, R3, V3 = 8, 9, 10, 11  # Third group (e.g., seniors)

# Initial conditions for the three groups using values from the Excel file
initial_conditions = [
    float(S_init[0]), float(I_init[0]), float(R_init[0]), float(V_init[0]),  # S1, I1, R1, V1 (Group 1)
    float(S_init[1]), float(I_init[1]), float(R_init[1]), float(V_init[1]),  # S2, I2, R2, V2 (Group 2)
    float(S_init[2]), float(I_init[2]), float(R_init[2]), float(V_init[2])   # S3, I3, R3, V3 (Group 3)
]
#%%
# SIRV model differential equations including quarantine dynamics and economic costs
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

# Economic cost calculations
def calculate_costs(I, V, D, t):
    # Medical cost
    medical_cost = (I[0] + I[1] + I[2]) * (m * C_M + h * C_H + i * C_I)
    
    # Vaccination cost
    vaccination_cost = (V[0] + V[1] + V[2]) * (VC + PC + LC)
    
    # Productivity loss cost
    # Group 1 affects Group 2 because parents have to take care of children
    productivity_cost = sum(I[j] * E[j] * D_in[j] for j in range(3))
    productivity_cost += I[0] * E[1] * D_in[1]  # Additional productivity loss for Group 2 due to Group 1
    
    # Funeral cost (if applicable)
    funeral_cost = D * FC if D > 0 else 0
    
    # Total cost
    total_cost = medical_cost + vaccination_cost + productivity_cost + funeral_cost
    
    return total_cost

#%%
# Time grid (in days)
t = np.linspace(0, time_span, int(time_span))

# Integrate the SIRV equations over time with quarantine logic
results = odeint(deriv, initial_conditions, t, args=(N, beta, beta_quarantine, quarantine_day, gamma, mu, W, a))

# Extract results for each group
S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = results.T

# Identify local maxima for the infected groups
def find_local_maxima(I):
    maxima_indices = argrelextrema(I, np.greater)[0]
    maxima_values = I[maxima_indices]
    return maxima_indices, maxima_values

maxima_I1_idx, maxima_I1_val = find_local_maxima(I1)
maxima_I2_idx, maxima_I2_val = find_local_maxima(I2)
maxima_I3_idx, maxima_I3_val = find_local_maxima(I3)

#%%
# Calculate economic costs over time
costs = []
for idx in range(len(t)):
    I = [I1[idx], I2[idx], I3[idx]]
    V = [V1[idx], V2[idx], V3[idx]]
    D = 0  # Assuming no death component in current implementation
    cost = calculate_costs(I, V, D, t[idx])
    costs.append(cost)

#%%
# Plot the data for each group, highlighting the local maxima for the infected compartments
plt.figure(figsize=(14, 16))
for i, (S, I, R, V, maxima_idx, maxima_val, group) in enumerate(zip(
        [S1, S2, S3], [I1, I2, I3], [R1, R2, R3], [V1, V2, V3],
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
    for idx, val in zip(maxima_idx, maxima_val):
        plt.annotate(f'({t[idx]:.1f}, {val:.1f})', (t[idx], val), textcoords="offset points", xytext=(0, 10), ha='center')
    
    plt.title(f'SIRV Dynamics for {group} with Quarantine Applied on Day {quarantine_day}')
    plt.xlabel('Days')
    plt.ylabel('Population')
    plt.legend()

# Plot total economic costs over time
plt.subplot(4, 1, 4)
plt.plot(t, costs, 'k', label='Total Economic Cost')
plt.title('Total Economic Cost Over Time')
plt.xlabel('Days')
plt.ylabel('Cost ($)')
plt.legend()

plt.tight_layout()
plt.show()

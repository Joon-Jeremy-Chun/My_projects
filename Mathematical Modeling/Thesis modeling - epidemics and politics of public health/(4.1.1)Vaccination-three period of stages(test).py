# -*- coding: utf-8 -*-
"""
Extended Model with Dynamic Vaccination Strategy (3.5)
Created on Tue Jan 26 21:25:57 2025

@author: joonc
"""

import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

##############################################################################
# 1) Function to load parameters from Excel
##############################################################################
def load_parameters(file_path):
    """
    Reads model parameters from an Excel file (Inputs.xlsx).
    Expects the following rows (adjust as needed for your file):
      - Row  4: recovery rates, gamma[0:3]
      - Row  6: maturity rates, mu[0:2]
      - Row  8: waning immunity rate, W
      - Row 12: time_span (integer/float)
      - Row 14: population_size (N) for 3 groups
      - Row 16: infectious_init (I) for 3 groups
      - Row 18: recovered_init (R) for 3 groups
      - Row 20: vaccinated_init (V) for 3 groups
    
    Then computes S_init = N - I_init - R_init - V_init
    """
    df = pd.read_excel(file_path, header=None)
    
    recovery_rates      = df.iloc[4,  0:3].values  # gamma
    maturity_rates      = df.iloc[6,  0:2].values  # mu
    waning_immunity_rate = df.iloc[8, 0]           # W
    time_span           = df.iloc[12, 0]
    
    # Read population size and initial states
    population_size   = df.iloc[14, 0:3].values
    infectious_init   = df.iloc[16, 0:3].values
    recovered_init    = df.iloc[18, 0:3].values
    vaccinated_init   = df.iloc[20, 0:3].values
    
    # Compute susceptible as: S = N - I - R - V
    susceptible_init = population_size - infectious_init - recovered_init - vaccinated_init
    
    return (recovery_rates, 
            maturity_rates, 
            waning_immunity_rate,
            time_span, 
            population_size,
            susceptible_init,
            infectious_init, 
            recovered_init, 
            vaccinated_init)

##############################################################################
# 2) Load parameters and data
##############################################################################
file_path = 'Inputs.xlsx'
gamma, mu, W, time_span, N, S_init, I_init, R_init, V_init = load_parameters(file_path)

# Load the transmission_rates matrix from the CSV file (adjust path if needed)
transmission_rates_csv_path = 'DataSets/Fitted_Beta_Matrix.csv'
beta = pd.read_csv(transmission_rates_csv_path).iloc[0:3, 1:4].values

##############################################################################
# 3) Define initial conditions for the ODE
##############################################################################
initial_conditions = [
    S_init[0], I_init[0], R_init[0], V_init[0],
    S_init[1], I_init[1], R_init[1], V_init[1],
    S_init[2], I_init[2], R_init[2], V_init[2]
]

##############################################################################
# 4) Define a (toggleable) vaccination rate
##############################################################################
# Toggle which line is active to switch scenarios:
#vaccination_rates = np.array([0.1, 0.1, 0.1])
#vaccination_rates = np.array([0.01, 0.01, 0.01])   # Nonzero vaccination
vaccination_rates = np.array([0.02, 0.02, 0.02])
#vaccination_rates = np.array([0, 0, 0]) # Zero vaccination

print("Using vaccination rates =", vaccination_rates)

# Define time thresholds for piecewise vaccination strategy
t1, t2 = 30, 60
B_timeP0, B_timeP1, B_timeP2 = 1.0, 1.0, 1.0  # multipliers for each phase

##############################################################################
# 5) Define the piecewise vaccination function
##############################################################################
def vaccination_strategy(t):
    """Defines time-dependent vaccination rates for each group."""
    if t < t1:
        return vaccination_rates * B_timeP0
    elif t1 <= t < t2:
        return vaccination_rates * B_timeP1
    else:
        return vaccination_rates * B_timeP2

##############################################################################
# 6) Define the Extended SIRV model with time-dependent vaccination
##############################################################################
def deriv(y, t, N, beta, gamma, mu, W):
    """
    ODE system for 3 groups:
      S, I, R, V compartments in each group,
      including aging (mu) from group1->group2, group2->group3,
      and waning immunity (W) returning R or V to S.
    """
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y
    
    # Get vaccination rates at time t
    a_t = vaccination_strategy(t)
    
    # Calculate forces of infection
    lambda1 = beta[0,0]*(I1/N[0]) + beta[0,1]*(I2/N[1]) + beta[0,2]*(I3/N[2])
    lambda2 = beta[1,0]*(I1/N[0]) + beta[1,1]*(I2/N[1]) + beta[1,2]*(I3/N[2])
    lambda3 = beta[2,0]*(I1/N[0]) + beta[2,1]*(I2/N[1]) + beta[2,2]*(I3/N[2])
    
    # Group 1
    dS1dt = -lambda1*S1 - (a_t[0])*S1 + W*R1 + W*V1
    dI1dt = lambda1*S1 - gamma[0]*I1 - mu[0]*I1
    dR1dt = gamma[0]*I1 - W*R1 - mu[0]*R1
    dV1dt = (a_t[0])*S1 - W*V1
    
    # Group 2
    dS2dt = -lambda2*S2 + mu[0]*S1 - (a_t[1])*S2 + W*R2 + W*V2
    dI2dt = lambda2*S2 + mu[0]*I1 - gamma[1]*I2 - mu[1]*I2
    dR2dt = gamma[1]*I2 + mu[0]*R1 - W*R2 - mu[1]*R2
    dV2dt = (a_t[1])*S2 - W*V2
    
    # Group 3
    dS3dt = -lambda3*S3 + mu[1]*S2 - (a_t[2])*S3 + W*R3 + W*V3
    dI3dt = lambda3*S3 + mu[1]*I2 - gamma[2]*I3
    dR3dt = gamma[2]*I3 + mu[1]*R2 - W*R3
    dV3dt = (a_t[2])*S3 - W*V3
    
    return (
        dS1dt, dI1dt, dR1dt, dV1dt,
        dS2dt, dI2dt, dR2dt, dV2dt,
        dS3dt, dI3dt, dR3dt, dV3dt
    )

##############################################################################
# 7) Solve the ODE system
##############################################################################
t = np.linspace(0, time_span, int(time_span))

dyn_results = odeint(deriv, initial_conditions, t, args=(N, beta, gamma, mu, W))

##############################################################################
# 8) Extract compartments for convenience
##############################################################################
S1, I1, R1, V1 = dyn_results[:, 0], dyn_results[:, 1], dyn_results[:, 2],  dyn_results[:, 3]
S2, I2, R2, V2 = dyn_results[:, 4], dyn_results[:, 5], dyn_results[:, 6],  dyn_results[:, 7]
S3, I3, R3, V3 = dyn_results[:, 8], dyn_results[:, 9], dyn_results[:, 10], dyn_results[:, 11]

I_total = I1 + I2 + I3

##############################################################################
# 9) Plot individual group figures with maxima
##############################################################################
fig_labels = ['Group 1 (Children)', 'Group 2 (Adults)', 'Group 3 (Seniors)']
for i, label in enumerate(fig_labels):
    # i-th infected = dyn_results[:, 1 + i*4]
    I_group = dyn_results[:, 1 + i*4]
    peak_idx = np.argmax(I_group)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, I_group, label=f'Infected {label}')
    plt.scatter(t[peak_idx], I_group[peak_idx], color='red', marker='o',
                label=f'Max: ({t[peak_idx]:.2f}, {I_group[peak_idx]:.2f})')
    plt.title(f'Infectious Population for {label} with Dynamic Vaccination')
    plt.xlabel('Days')
    plt.ylabel('Number of Infected Individuals')
    plt.legend()
    plt.grid()
    plt.show()

# Plot total infected
plt.figure(figsize=(10, 6))
peak_idx_total = np.argmax(I_total)
plt.plot(t, I_total, label='Total Infected')
plt.scatter(t[peak_idx_total], I_total[peak_idx_total],
            color='red', marker='o',
            label=f'Max: ({t[peak_idx_total]:.2f}, {I_total[peak_idx_total]:.2f})')
plt.title('Total Infectious Population with Dynamic Vaccination')
plt.xlabel('Days')
plt.ylabel('Number of Infected Individuals')
plt.legend()
plt.grid()
plt.show()

##############################################################################
# 10) Plot all groups in one figure with maxima
##############################################################################
plt.figure(figsize=(10, 6))
plt.plot(t, I1, label='Children', color='blue')
plt.scatter(t[np.argmax(I1)], max(I1), color='red', marker='o',
            label=f'Max (Children): ({t[np.argmax(I1)]:.2f}, {max(I1):.2f})')

plt.plot(t, I2, label='Adults', color='green')
plt.scatter(t[np.argmax(I2)], max(I2), color='red', marker='o',
            label=f'Max (Adults): ({t[np.argmax(I2)]:.2f}, {max(I2):.2f})')

plt.plot(t, I3, label='Seniors', color='red')
plt.scatter(t[np.argmax(I3)], max(I3), color='red', marker='o',
            label=f'Max (Seniors): ({t[np.argmax(I3)]:.2f}, {max(I3):.2f})')

plt.title('Comparison of Infectious Populations Across Groups')
plt.xlabel('Days')
plt.ylabel('Number of Infected Individuals')
plt.legend()
plt.grid()
plt.show()

##############################################################################
# 11) Print peak values and days
##############################################################################
print("Peak Infections and Days:")
print(f"Children: Peak = {max(I1):.2f}, Day = {t[np.argmax(I1)]:.2f}")
print(f"Adults:   Peak = {max(I2):.2f}, Day = {t[np.argmax(I2)]:.2f}")
print(f"Seniors:  Peak = {max(I3):.2f}, Day = {t[np.argmax(I3)]:.2f}")
print(f"Total:    Peak = {max(I_total):.2f}, Day = {t[np.argmax(I_total)]:.2f}")

##############################################################################
# 12) Plot vaccination rates over time
##############################################################################
t_plot = np.linspace(0, time_span, int(time_span))
vaccination_rates_over_time = np.array([vaccination_strategy(time) 
                                        for time in t_plot])

vaccination_children = vaccination_rates_over_time[:, 0]
vaccination_adults   = vaccination_rates_over_time[:, 1]
vaccination_seniors  = vaccination_rates_over_time[:, 2]

plt.figure(figsize=(10, 6))
plt.plot(t_plot, vaccination_children, label="Children", color='blue')
plt.plot(t_plot, vaccination_adults,   label="Adults",   color='green')
plt.plot(t_plot, vaccination_seniors,  label="Seniors",  color='red')
plt.title("Vaccination Strategy Over Time")
plt.xlabel("Days")
plt.ylabel("Vaccination Rate")
plt.legend()
plt.grid()
plt.show()

##############################################################################
# (Optional) Plot the Vaccinated compartments for each group
##############################################################################
plt.figure(figsize=(10,6))
plt.plot(t, V1, label='V1 - Children')
plt.plot(t, V2, label='V2 - Adults')
plt.plot(t, V3, label='V3 - Seniors')
plt.xlabel("Days")
plt.ylabel("Number Vaccinated")
plt.title("Vaccinated Compartments Over Time")
plt.legend()
plt.grid()
plt.show()

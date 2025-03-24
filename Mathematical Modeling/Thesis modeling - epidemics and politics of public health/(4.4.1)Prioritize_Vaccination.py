# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 09:48:07 2025

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
import os

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
# 2. TIME SEGMENTS & GLOBAL VACCINATION ARRAYS
# -----------------------------
# Segment boundaries (in days)
t_a_end = 30      # Period a
t_b_end = 60      # Period b
t_c_end = 180     # Period c
# Period d starts after t_c_end

# Global arrays for dynamic vaccination rates; these will be updated for each example.
vaccination_rates_dynamic_a = np.array([0.0, 0.0, 0.0])
vaccination_rates_dynamic_b = np.array([0.0, 0.0, 0.0])
vaccination_rates_dynamic_c = np.array([0.0, 0.0, 0.0])
vaccination_rates_dynamic_d = np.array([0.0, 0.0, 0.0])

# Manual vaccination rates (if dynamic strategy is not desired)
vaccination_rates_manual = np.array([0.0, 0.0, 0.0])

def vaccination_strategy(t, use_dynamic=True):
    """
    Returns the vaccination rate vector at time t.
    If use_dynamic is True, uses the four-phase dynamic strategy;
    otherwise, returns the manual vaccination rates.
    """
    if use_dynamic:
        if t < t_a_end:
            return vaccination_rates_dynamic_a
        elif t < t_b_end:
            return vaccination_rates_dynamic_b
        elif t < t_c_end:
            return vaccination_rates_dynamic_c
        else:
            return vaccination_rates_dynamic_d
    else:
        return vaccination_rates_manual

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
    
    # Force of infection for each group (with cross-group mixing)
    lambda1 = beta[0, 0] * I1/N[0] + beta[0, 1] * I2/N[1] + beta[0, 2] * I3/N[2]
    lambda2 = beta[1, 0] * I1/N[0] + beta[1, 1] * I2/N[1] + beta[1, 2] * I3/N[2]
    lambda3 = beta[2, 0] * I1/N[0] + beta[2, 1] * I2/N[1] + beta[2, 2] * I3/N[2]
    
    # Group 1 (Children)
    dS1dt = -lambda1 * S1 - a_t[0] * S1 + W * R1 + W * V1
    dI1dt = lambda1 * S1 - gamma[0] * I1
    dR1dt = gamma[0] * I1 - W * R1
    dV1dt = a_t[0] * S1 - W * V1
    
    # Group 2 (Adults)
    dS2dt = -lambda2 * S2 - a_t[1] * S2 + W * R2 + W * V2
    dI2dt = lambda2 * S2 - gamma[1] * I2
    dR2dt = gamma[1] * I2 - W * R2
    dV2dt = a_t[1] * S2 - W * V2
    
    # Group 3 (Seniors)
    dS3dt = -lambda3 * S3 - a_t[2] * S3 + W * R3 + W * V3
    dI3dt = lambda3 * S3 - gamma[2] * I3
    dR3dt = gamma[2] * I3 - W * R3
    dV3dt = a_t[2] * S3 - W * V3
    
    return [
        dS1dt, dI1dt, dR1dt, dV1dt,
        dS2dt, dI2dt, dR2dt, dV2dt,
        dS3dt, dI3dt, dR3dt, dV3dt
    ]

# -----------------------------
# 4. UTILITY FUNCTION: RUN SIMULATION & PLOT
# -----------------------------
def run_example_and_plot(example_label, figure_title,
                         vacc_a, vacc_b, vacc_c, vacc_d,
                         save_figure=True):
    """
    1) Updates the global vaccination arrays for each period (a, b, c, d).
    2) Runs the simulation with dynamic vaccination.
    3) Plots I(t) for each age group plus total infections.
    4) Adds an overall figure title and saves the figure to 'Figures/'.
    """
    global vaccination_rates_dynamic_a, vaccination_rates_dynamic_b
    global vaccination_rates_dynamic_c, vaccination_rates_dynamic_d
    
    # Update global vaccination arrays
    vaccination_rates_dynamic_a = np.array(vacc_a)
    vaccination_rates_dynamic_b = np.array(vacc_b)
    vaccination_rates_dynamic_c = np.array(vacc_c)
    vaccination_rates_dynamic_d = np.array(vacc_d)
    
    # Create time grid and run ODE solver
    t = np.linspace(0, time_span, int(time_span))
    results = odeint(deriv, initial_conditions, t, args=(N, beta, gamma, mu, W, True))
    
    # Extract infected individuals for each group
    I_0_20 = results[:, 1]   # Group 1 (Children)
    I_20_49 = results[:, 5]   # Group 2 (Adults)
    I_50_80 = results[:, 9]   # Group 3 (Seniors)
    I_total = I_0_20 + I_20_49 + I_50_80
    
    # Create the figure and subplots
    plt.figure(figsize=(12, 10))
    plt.suptitle(figure_title, fontsize=16, y=0.98)
    
    plt.subplot(2, 2, 1)
    plt.plot(t, I_0_20, label='I(t) Age 0-19')
    plt.xlabel('Time (days)')
    plt.ylabel('Infected Individuals')
    plt.title('Infections Age 0-19')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(t, I_20_49, label='I(t) Age 20-49')
    plt.xlabel('Time (days)')
    plt.ylabel('Infected Individuals')
    plt.title('Infections Age 20-49')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(t, I_50_80, label='I(t) Age 50-80+')
    plt.xlabel('Time (days)')
    plt.ylabel('Infected Individuals')
    plt.title('Infections Age 50-80+')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(t, I_total, label='I(t) Total', color='black')
    plt.xlabel('Time (days)')
    plt.ylabel('Infected Individuals')
    plt.title('Total Infections')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure in the Figures folder
    if save_figure:
        os.makedirs('Figures', exist_ok=True)
        figure_path = f'Figures/{example_label}.png'
        plt.savefig(figure_path, dpi=150)
        print(f"Saved figure to {figure_path}")
    
    plt.show()

# -----------------------------
# 5. RUN THREE EXAMPLES
# -----------------------------
# Example 1: Children prioritized initially
run_example_and_plot(
    example_label='Children Prioritized',
    figure_title='Children Prioritized in first 60days',
    vacc_a=[0.015, 0.0, 0.0],    # Period a
    vacc_b=[0.015, 0.0, 0.0],    # Period b
    vacc_c=[0.015, 0.015, 0.015],# Period c
    vacc_d=[0.015, 0.015, 0.015] # Period d
)

# Example 2: Adults prioritized initially
run_example_and_plot(
    example_label='Adults Prioritized',
    figure_title='Adults Prioritized in first 60days',
    vacc_a=[0.0, 0.015, 0.0],
    vacc_b=[0.0, 0.015, 0.0],
    vacc_c=[0.015, 0.015, 0.015],
    vacc_d=[0.015, 0.015, 0.015]
)

# Example 3: Seniors prioritized initially
run_example_and_plot(
    example_label='Seniors Prioritized',
    figure_title='Seniors Prioritized in first 60days',
    vacc_a=[0.0, 0.0, 0.015],
    vacc_b=[0.0, 0.0, 0.015],
    vacc_c=[0.015, 0.015, 0.015],
    vacc_d=[0.015, 0.015, 0.015]
)

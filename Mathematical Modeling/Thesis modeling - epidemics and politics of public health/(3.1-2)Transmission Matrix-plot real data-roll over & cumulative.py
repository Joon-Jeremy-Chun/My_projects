# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 06:27:49 2024

@author: joonc
"""

import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Step 1: Load Dataset and Inputs Values

# Load the data for the model inputs
file_path_inputs = 'Inputs.xlsx'
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
    susceptible_init = df.iloc[14, 0:3].values
    infectious_init = df.iloc[16, 0:3].values
    recovered_init = df.iloc[18, 0:3].values
    vaccinated_init = df.iloc[20, 0:3].values

    return (transmission_rates, recovery_rates, maturity_rates, waning_immunity_rate,
            vaccination_rates, time_span, population_size, susceptible_init, infectious_init,
            recovered_init, vaccinated_init)

(beta, gamma, mu, W, a, time_span, N, S_init, I_init, R_init, V_init) = load_parameters(file_path_inputs)

# Define initial conditions for the SIRV model
initial_conditions = [
    S_init[0], I_init[0], R_init[0], V_init[0],  # Group 1
    S_init[1], I_init[1], R_init[1], V_init[1],  # Group 2
    S_init[2], I_init[2], R_init[2], V_init[2]   # Group 3
]

# Define the SIRV model differential equations
def deriv(y, t, N, beta, gamma, mu, W, a):
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y

    # Force of infection (\u03bb_i) for each group
    lambda1 = beta[0, 0] * I1/N[0] + beta[0, 1] * I2/N[1] + beta[0, 2] * I3/N[2]
    lambda2 = beta[1, 0] * I1/N[0] + beta[1, 1] * I2/N[1] + beta[1, 2] * I3/N[2]
    lambda3 = beta[2, 0] * I1/N[0] + beta[2, 1] * I2/N[1] + beta[2, 2] * I3/N[2]

    # Differential equations
    dS1dt = -lambda1 * S1 - a[0] / N[0] * S1 + W * R1 + W * V1
    dI1dt = lambda1 * S1 - gamma[0] * I1 - mu[0] * I1
    dR1dt = gamma[0] * I1 - W * R1 - mu[0] * R1
    dV1dt = a[0] / N[0] * S1 - W * V1

    dS2dt = -lambda2 * S2 + mu[0] * S1 - a[1] / N[1] * S2 + W * R2 + W * V2
    dI2dt = lambda2 * S2 + mu[0] * I1 - gamma[1] * I2 - mu[1] * I2
    dR2dt = gamma[1] * I2 + mu[0] * R1 - W * R2 - mu[1] * R2
    dV2dt = a[1] / N[1] * S2 - W * V2

    dS3dt = -lambda3 * S3 + mu[1] * S2 - a[2] / N[2] * S3 + W * R3 + W * V3
    dI3dt = lambda3 * S3 + mu[1] * I2 - gamma[2] * I3
    dR3dt = gamma[2] * I3 + mu[1] * R2 - W * R3
    dV3dt = a[2] / N[2] * S3 - W * V3

    return dS1dt, dI1dt, dR1dt, dV1dt, dS2dt, dI2dt, dR2dt, dV2dt, dS3dt, dI3dt, dR3dt, dV3dt

# Simulate the model over the time span
t = np.linspace(0, time_span, int(time_span))
results = odeint(deriv, initial_conditions, t, args=(N, beta, gamma, mu, W, a))

# Step 2: Plot Results

# Extract the results for each group
S1, I1, R1, V1 = results[:, 0], results[:, 1], results[:, 2], results[:, 3]
S2, I2, R2, V2 = results[:, 4], results[:, 5], results[:, 6], results[:, 7]
S3, I3, R3, V3 = results[:, 8], results[:, 9], results[:, 10], results[:, 11]

# Total compartments
S_total = S1 + S2 + S3
I_total = I1 + I2 + I3
R_total = R1 + R2 + R3
V_total = V1 + V2 + V3

# Define Method 1: Loo-over with Average Recovery Date
def method1_new_infected(I, gamma, t):
    avg_recovery_duration = 1 / gamma  # Average recovery duration
    I_method1 = np.zeros_like(I)
    for day in range(len(t)):
        if day < avg_recovery_duration:
            I_method1[day] = np.sum(I[:day+1])
        else:
            I_method1[day] = np.sum(I[day-int(avg_recovery_duration):day+1])
    return I_method1

# Define Method 2: Cumulative with Daily Recovery Transition
def method2_cumulative(I, gamma):
    I_method2 = np.zeros_like(I)
    I_current = 0
    for day in range(len(I)):
        I_current += I[day] - gamma * I_current  # Update current infected
        I_method2[day] = I_current
    return I_method2

# Calculate and Plot for Each Group
def plot_comparison(t, I, gamma, group_label):
    I_method1 = method1_new_infected(I, gamma, t)
    I_method2 = method2_cumulative(I, gamma)

    plt.figure(figsize=(10, 6))
    plt.plot(t, I_method1, label='Method 1: Loo-over', linestyle='--')
    plt.plot(t, I_method2, label='Method 2: Cumulative Transition', linestyle='-')
    plt.title(f'Comparison of Methods for {group_label}')
    plt.xlabel('Days')
    plt.ylabel('Infectious Population')
    plt.legend()
    plt.grid()
    plt.show()

# Plot for Group 1 (Children)
plot_comparison(t, I1, gamma[0], "Group 1 (Children)")

# Plot for Group 2 (Adults)
plot_comparison(t, I2, gamma[1], "Group 2 (Adults)")

# Plot for Group 3 (Seniors)
plot_comparison(t, I3, gamma[2], "Group 3 (Seniors)")

# Plot for Total Population
plot_comparison(t, I_total, np.mean(gamma), "Total Population")

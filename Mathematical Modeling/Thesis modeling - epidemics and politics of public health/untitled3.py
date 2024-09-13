# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 13:23:32 2024

@author: joonc
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Total populations for each group
N = [2000, 3000, 3000]

# Sigmoid function for vaccination rate with separate parameters for each group
def sigmoid(t, t0, k, a):
    return a / (1 + np.exp(-k * (t - t0)))

# Vaccination rate function for each group
def vaccination_rate(t, t0, a, k):
    return sigmoid(t, t0, k, a)

# SIRV model differential equations including vaccinated compartments
def deriv(y, t, N, beta, gamma, mu, W, t0, a1, a2, a3, k1, k2, k3):
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y
    
    # Force of infection (λ_i) for each group
    lambda1 = beta[0, 0] * I1/N[0] + beta[0, 1] * I2/N[1] + beta[0, 2] * I3/N[2]
    lambda2 = beta[1, 0] * I1/N[0] + beta[1, 1] * I2/N[1] + beta[1, 2] * I3/N[2]
    lambda3 = beta[2, 0] * I1/N[0] + beta[2, 1] * I2/N[1] + beta[2, 2] * I3/N[2]
    
    # Time-dependent vaccination rates using separate sigmoid functions for each group
    a0_1 = vaccination_rate(t, t0, a1, k1)
    a0_2 = vaccination_rate(t, t0, a2, k2)
    a0_3 = vaccination_rate(t, t0, a3, k3)
    
    # Differential equations for each compartment
    # Group 1 (e.g., children)
    dS1dt = -lambda1 * S1 - a0_1 * S1 + W * R1 + W * V1
    dI1dt = lambda1 * S1 - gamma[0] * I1 - mu[0] * I1
    dR1dt = gamma[0] * I1 - W * R1 - mu[0] * R1
    dV1dt = a0_1 * S1 - W * V1
    
    # Group 2 (e.g., adults)
    dS2dt = -lambda2 * S2 + mu[0] * S1 - a0_2 * S2 + W * R2 + W * V2
    dI2dt = lambda2 * S2 + mu[0] * I1 - gamma[1] * I2 - mu[1] * I2
    dR2dt = gamma[1] * I2 + mu[0] * R1 - W * R2 - mu[1] * R2
    dV2dt = a0_2 * S2 - W * V2
    
    # Group 3 (e.g., seniors)
    dS3dt = -lambda3 * S3 + mu[1] * S2 - a0_3 * S3 + W * R3 + W * V3
    dI3dt = lambda3 * S3 + mu[1] * I2 - gamma[2] * I3
    dR3dt = gamma[2] * I3 + mu[1] * R2 - W * R3
    dV3dt = a0_3 * S3 - W * V3
    
    return dS1dt, dI1dt, dR1dt, dV1dt, dS2dt, dI2dt, dR2dt, dV2dt, dS3dt, dI3dt, dR3dt, dV3dt

# Initial conditions for the three groups (including vaccinated compartments)
initial_conditions = [
    2000 - 0.0001 * 2000, 0.0001 * 2000, 0, 0,  # S1, I1, R1, V1 (Group 1)
    3000 - 0.0001 * 3000, 0.0001 * 3000, 0, 0,  # S2, I2, R2, V2 (Group 2)
    3000 - 0.0001 * 3000, 0.0001 * 3000, 0, 0   # S3, I3, R3, V3 (Group 3)
]

# Time grid (in days)
t = np.linspace(0, 300, 3000)

# Transmission rates matrix (β_ij) and other parameters
beta = np.array([
    [0.18, 0.06, 0.03],  # β11, β12, β13 (from I1, I2, I3 to S1)
    [0.04, 0.12, 0.04],  # β21, β22, β23 (from I1, I2, I3 to S2)
    [0.03, 0.04, 0.07]   # β31, β32, β33 (from I1, I2, I3 to S3)
])

# Recovery rates γ_i for the three groups
gamma = [0.12, 0.1, 0.08]

# Maturity rates μ1, μ2 between groups
mu = [0.0001, 0.0002]

# Waning immunity rate W for recovered and vaccinated individuals returning to susceptible
W = 0.00005

# Vaccination parameters: t0 (time of introduction), a1, a2, a3 (max vaccination rates), and k1, k2, k3 (steepness)
t0 = 50  # Time at which vaccination starts
a1, a2, a3 = 0.002, 0.001, 0.005  # Maximum vaccination rates for each group
k1, k2, k3 = 0.1, 0.2, 0.3  # Steepness of the sigmoid function for each group

# Integrate the SIRV equations over time
results = odeint(deriv, initial_conditions, t, args=(N, beta, gamma, mu, W, t0, a1, a2, a3, k1, k2, k3))

# Extract results for each group
S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = results.T

# Plot the data for each group
plt.figure(figsize=(14, 12))
for i, (S, I, R, V, group) in enumerate(zip([S1, S2, S3], [I1, I2, I3], [R1, R2, R3], [V1, V2, V3], ['Group 1', 'Group 2', 'Group 3']), start=1):
    plt.subplot(3, 1, i)
    plt.plot(t, S, 'b', label=f'Susceptible ({group})')
    plt.plot(t, I, 'r', label=f'Infected ({group})')
    plt.plot(t, R, 'g', label=f'Recovered ({group})')
    plt.plot(t, V, 'm', label=f'Vaccinated ({group})')
    plt.title(f'SIRV Dynamics for {group}')
    plt.xlabel('Days')
    plt.ylabel('Population')
    plt.legend()

plt.tight_layout()
plt.show()

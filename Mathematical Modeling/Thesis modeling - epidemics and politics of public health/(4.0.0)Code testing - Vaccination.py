# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 22:22:21 2025

@author: joonc
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

##############################################################################
# 1) Define parameters and an easy toggle for vaccination rates.
##############################################################################
use_zero_vaccination = False  # <--- Toggle this to True/False to compare

if use_zero_vaccination:
    vaccination_rates = np.array([0.0, 0.0, 0.0])
else:
    vaccination_rates = np.array([1.0, 1.0, 1.0])

print("Using vaccination rates =", vaccination_rates)

# Example fixed parameters (replace with your actual or loaded values)
N = np.array([1_000, 2_000, 500])  # population sizes
gamma = np.array([0.1, 0.1, 0.1])  # recovery rates
mu = np.array([0.01, 0.01])        # aging rates: children->adults, adults->seniors
W = 0.001                          # waning rate
time_span = 200                    # total days to simulate

# Transmission matrix (just an example 3x3)
beta = np.array([
    [0.3, 0.1, 0.05],
    [0.1, 0.3, 0.05],
    [0.05,0.05, 0.2]
])

# Initial conditions
S_init = np.array([900, 1500, 400])  # susceptible
I_init = np.array([100,  500,  100]) # infected
R_init = np.array([0,    0,    0])   # recovered
V_init = np.array([0,    0,    0])   # vaccinated

##############################################################################
# 2) Define the piecewise multipliers for vaccination over time (if desired).
##############################################################################
t1, t2 = 30, 60
B_timeP0, B_timeP1, B_timeP2 = 1.0, 1.0, 1.0  # all = 1 for simplicity

def vaccination_strategy(t):
    """Time-dependent vaccination multiplier."""
    if t < t1:
        return vaccination_rates * B_timeP0
    elif t1 <= t < t2:
        return vaccination_rates * B_timeP1
    else:
        return vaccination_rates * B_timeP2

##############################################################################
# 3) Define the ODE system
##############################################################################
def deriv(y, t, N, beta, gamma, mu, W):
    """
    SIRV model with aging (mu) and waning (W) for 3 groups.
    State order: S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3
    """
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y
    a_t = vaccination_strategy(t)

    # Forces of infection
    lambda1 = beta[0,0]*I1/N[0] + beta[0,1]*I2/N[1] + beta[0,2]*I3/N[2]
    lambda2 = beta[1,0]*I1/N[0] + beta[1,1]*I2/N[1] + beta[1,2]*I3/N[2]
    lambda3 = beta[2,0]*I1/N[0] + beta[2,1]*I2/N[1] + beta[2,2]*I3/N[2]

    # Group 1
    dS1dt = -lambda1*S1 - (a_t[0]/N[0])*S1 + W*R1 + W*V1
    dI1dt = lambda1*S1 - gamma[0]*I1 - mu[0]*I1
    dR1dt = gamma[0]*I1 - W*R1 - mu[0]*R1
    dV1dt = (a_t[0]/N[0])*S1 - W*V1

    # Group 2
    dS2dt = -lambda2*S2 + mu[0]*S1 - (a_t[1]/N[1])*S2 + W*R2 + W*V2
    dI2dt = lambda2*S2 + mu[0]*I1 - gamma[1]*I2 - mu[1]*I2
    dR2dt = gamma[1]*I2 + mu[0]*R1 - W*R2 - mu[1]*R2
    dV2dt = (a_t[1]/N[1])*S2 - W*V2

    # Group 3
    dS3dt = -lambda3*S3 + mu[1]*S2 - (a_t[2]/N[2])*S3 + W*R3 + W*V3
    dI3dt = lambda3*S3 + mu[1]*I2 - gamma[2]*I3
    dR3dt = gamma[2]*I3 + mu[1]*R2 - W*R3
    dV3dt = (a_t[2]/N[2])*S3 - W*V3

    return [dS1dt, dI1dt, dR1dt, dV1dt,
            dS2dt, dI2dt, dR2dt, dV2dt,
            dS3dt, dI3dt, dR3dt, dV3dt]

##############################################################################
# 4) Solve the system
##############################################################################
initial_conditions = [
    S_init[0], I_init[0], R_init[0], V_init[0],
    S_init[1], I_init[1], R_init[1], V_init[1],
    S_init[2], I_init[2], R_init[2], V_init[2]
]

t = np.linspace(0, time_span, int(time_span)+1)
solution = odeint(deriv, initial_conditions, t, args=(N, beta, gamma, mu, W))

# Unpack the solution for easier plotting
S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = solution.T

##############################################################################
# 5) Plot infected and vaccinated compartments (to SEE the difference)
##############################################################################
plt.figure(figsize=(10,6))

plt.plot(t, I1, label="I1 (Children)")
plt.plot(t, I2, label="I2 (Adults)")
plt.plot(t, I3, label="I3 (Seniors)")

plt.title("Infected curves")
plt.xlabel("Days")
plt.ylabel("Number Infected")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,6))

plt.plot(t, V1, label="V1 (Children)")
plt.plot(t, V2, label="V2 (Adults)")
plt.plot(t, V3, label="V3 (Seniors)")

plt.title("Vaccinated curves")
plt.xlabel("Days")
plt.ylabel("Number Vaccinated")
plt.legend()
plt.grid(True)
plt.show()

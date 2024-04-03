# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 21:19:39 2024

@author: joonc
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define parameters
beta = 0.5  # Transmission rate
gamma = 0.1  # Recovery rate
age_groups = 2  # For simplicity, consider two age groups
N = 1000  # Total population

# Initial conditions: S0, I0, R0 for each age group
S0 = [800, 150]  # Susceptible population in each age group
I0 = [10, 40]  # Infected population in each age group
R0 = [N - S0[i] - I0[i] for i in range(age_groups)]  # Recovered population

# Define the SIR model equations
def sir_model(t, y):
    S = y[:age_groups]
    I = y[age_groups:2*age_groups]
    R = y[2*age_groups:]
    dSdt = [-beta * S[i] * I[i] / N for i in range(age_groups)]
    dIdt = [beta * S[i] * I[i] / N - gamma * I[i] for i in range(age_groups)]
    dRdt = [gamma * I[i] for i in range(age_groups)]
    return dSdt + dIdt + dRdt

# Solve the SIR model
t_span = [0, 100]  # Time span for the simulation
y0 = S0 + I0 + R0  # Initial conditions for all compartments
sol = solve_ivp(sir_model, t_span, y0, method='RK45', t_eval=np.linspace(t_span[0], t_span[1], 100))

# Plotting
plt.plot(sol.t, sol.y[:age_groups].T)
plt.xlabel('Time')
plt.ylabel('Susceptible Population')
plt.title('SIR Model with Age Groups - Susceptibles')
plt.legend(['Age group 1', 'Age group 2'])
plt.figure()

plt.plot(sol.t, sol.y[age_groups:2*age_groups].T)
plt.xlabel('Time')
plt.ylabel('Infected Population')
plt.title('SIR Model with Age Groups - Infected')
plt.legend(['Age group 1', 'Age group 2'])
plt.show()
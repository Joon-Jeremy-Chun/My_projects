# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 17:23:42 2024

@author: joonc
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol

# Group identifiers for zero-based indexing
HIGHLY_CAUTIOUS, MODERATELY_CAUTIOUS, LOW_CAUTIOUS = 0, 1, 2

# Define the deriv function as before
def deriv(y, t, N, beta, gamma, delta, a1, a2, b1, b2):
    SH, SM, SL, I, TI = y

    total_N = sum(N)
    proportion_H = 1/6
    proportion_M = 2/6
    proportion_L = 3/6

    dSHdt = -beta[HIGHLY_CAUTIOUS] * SH * I / N[HIGHLY_CAUTIOUS] + delta * TI * proportion_H \
            + b1 * SM - a1 * SH
    dSMdt = -beta[MODERATELY_CAUTIOUS] * SM * I / N[MODERATELY_CAUTIOUS] + delta * TI * proportion_M \
            + a1 * SH + b2 * SL - (a2 + b1) * SM
    dSLdt = -beta[LOW_CAUTIOUS] * SL * I / N[LOW_CAUTIOUS] + delta * TI * proportion_L \
            + a2 * SM - b2 * SL
    dIdt = (beta[HIGHLY_CAUTIOUS] * SH * I / N[HIGHLY_CAUTIOUS] +
            beta[MODERATELY_CAUTIOUS] * SM * I / N[MODERATELY_CAUTIOUS] +
            beta[LOW_CAUTIOUS] * SL * I / N[LOW_CAUTIOUS]) - gamma * I
    dTIdt = gamma * I - delta * TI

    return [dSHdt, dSMdt, dSLdt, dIdt, dTIdt]

# Define the initial conditions and parameters
initial_conditions = {
    HIGHLY_CAUTIOUS: {'S0': 800, 'I0': 1, 'TI0': 0},  
    MODERATELY_CAUTIOUS: {'S0': 150, 'I0': 1, 'TI0': 0},  
    LOW_CAUTIOUS: {'S0': 50, 'I0': 1, 'TI0': 0}  
}

# Transmission rates for each group
beta = [0.21, 0.22, 0.23]

# Recovery rate
gamma = 0.2

# Immunity loss rate
delta = 0.001  # Approximately corresponding to 8 months immunity duration

# Transition rates between groups
a1 = 0.2  # Caution Relaxation Rate (High to Moderate)
a2 = 0.2  # Caution Relaxation Rate (Moderate to Low)
b1 = 0.05  # Caution Intensification Rate (Moderate to High)
b2 = 0.05  # Caution Intensification Rate (Low to Moderate)

N = [initial_conditions[HIGHLY_CAUTIOUS]['S0'],
     initial_conditions[MODERATELY_CAUTIOUS]['S0'],
     initial_conditions[LOW_CAUTIOUS]['S0']]

# Initial conditions vector
y0 = [initial_conditions[HIGHLY_CAUTIOUS]['S0'], 
      initial_conditions[MODERATELY_CAUTIOUS]['S0'], 
      initial_conditions[LOW_CAUTIOUS]['S0'],
      initial_conditions[HIGHLY_CAUTIOUS]['I0'] + initial_conditions[MODERATELY_CAUTIOUS]['I0'] + initial_conditions[LOW_CAUTIOUS]['I0'], 
      initial_conditions[HIGHLY_CAUTIOUS]['TI0'] + initial_conditions[MODERATELY_CAUTIOUS]['TI0'] + initial_conditions[LOW_CAUTIOUS]['TI0']]

# Time grid (in days)
t = np.linspace(0, 300, 1600)

# Define the problem for SALib
problem = {
    'num_vars': 7,
    'names': ['beta1', 'beta2', 'beta3', 'gamma', 'delta', 'a1', 'a2'],
    'bounds': [
        [0.15, 0.25],
        [0.15, 0.25],
        [0.15, 0.25],
        [0.1, 0.3],
        [0.0005, 0.002],
        [0.1, 0.3],
        [0.1, 0.3]
    ]
}

# Generate samples
param_values = saltelli.sample(problem, 1000)

# Run the model for each sample
Y = np.zeros([param_values.shape[0]])

for i, params in enumerate(param_values):
    beta = params[0:3]
    gamma = params[3]
    delta = params[4]
    a1 = params[5]
    a2 = params[6]
    b1 = 0.05  # Fixed for simplicity
    b2 = 0.05  # Fixed for simplicity
    args = (N, beta, gamma, delta, a1, a2, b1, b2)
    ret = odeint(deriv, y0, t, args=args)
    SH, SM, SL, I, TI = ret.T
    Y[i] = np.max(I)  # For example, take the maximum number of infected individuals

# Perform Sobol sensitivity analysis
Si = sobol.analyze(problem, Y)

# Print the results
print("First-order sensitivity indices:")
print(Si['S1'])
print("Total sensitivity indices:")
print(Si['ST'])

# Plotting results
plt.figure(figsize=(10, 6))
plt.bar(problem['names'], Si['S1'], yerr=Si['S1_conf'], align='center', alpha=0.5, ecolor='black', capsize=10)
plt.xlabel('Parameters')
plt.ylabel('Sensitivity Index')
plt.title('First-order Sensitivity Analysis')
plt.show()


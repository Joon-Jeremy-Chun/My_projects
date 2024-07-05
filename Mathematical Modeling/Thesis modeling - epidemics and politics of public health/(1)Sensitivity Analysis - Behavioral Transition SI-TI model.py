# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 20:32:25 2024

@author: joonc
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

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

# Sensitivity analysis function
def sensitivity_analysis(param_name, param_values):
    results = []
    global beta, gamma, delta, a1, a2, b1, b2  # Use global variables for parameters
    for value in param_values:
        if param_name == 'beta':
            beta = value
        elif param_name == 'gamma':
            gamma = value
        elif param_name == 'delta':
            delta = value
        elif param_name == 'a1':
            a1 = value
        elif param_name == 'a2':
            a2 = value
        elif param_name == 'b1':
            b1 = value
        elif param_name == 'b2':
            b2 = value
        
        # Integrate the SIR equations over the time grid
        args = (N, beta, gamma, delta, a1, a2, b1, b2)
        ret = odeint(deriv, y0, t, args=args)
        SH, SM, SL, I, TI = ret.T
        results.append((t, SH, SM, SL, I, TI))
        
    return results

# Example usage
param_name = 'beta'
param_values = [
    [0.18, 0.19, 0.20],
    [0.21, 0.22, 0.23],
    [0.24, 0.25, 0.26]
]
results = sensitivity_analysis(param_name, param_values)

# Plotting results
plt.figure(figsize=(10, 6))
for i, result in enumerate(results):
    t, SH, SM, SL, I, TI = result
    plt.plot(t, I, label=f'{param_name}={param_values[i]}')
plt.xlabel('Days')
plt.ylabel('Infected Population')
plt.title(f'Sensitivity Analysis: Effect of {param_name}')
plt.legend()
plt.grid()
plt.show()

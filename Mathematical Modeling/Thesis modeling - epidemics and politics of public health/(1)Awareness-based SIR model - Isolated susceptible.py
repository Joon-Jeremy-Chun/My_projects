# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 20:57:51 2024

@author: joonc
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Group identifiers for zero-based indexing
HIGHLY_CAUTIOUS, MODERATELY_CAUTIOUS, LOW_CAUTIOUS = 0, 1, 2

def deriv(y, t, N, beta, gamma):
    """
    Compute the derivatives for the SIR model with different caution levels.

    Parameters:
    y (list): List of S, I, R values for each group.
    t (float): Time.
    N (list): Total population for each group.
    beta (list): Transmission rates for each group.
    gamma (float): Recovery rate for the infected group.

    Returns:
    list: List of derivatives for S, I, R values.
    """
    SH, SM, SL, I, R = y

    dSHdt = -beta[HIGHLY_CAUTIOUS] * SH * I / N[HIGHLY_CAUTIOUS]
    dSMdt = -beta[MODERATELY_CAUTIOUS] * SM * I / N[MODERATELY_CAUTIOUS]
    dSLdt = -beta[LOW_CAUTIOUS] * SL * I / N[LOW_CAUTIOUS]
    dIdt = (beta[HIGHLY_CAUTIOUS] * SH * I / N[HIGHLY_CAUTIOUS] +
            beta[MODERATELY_CAUTIOUS] * SM * I / N[MODERATELY_CAUTIOUS] +
            beta[LOW_CAUTIOUS] * SL * I / N[LOW_CAUTIOUS]) - gamma * I
    dRdt = gamma * I

    return [dSHdt, dSMdt, dSLdt, dIdt, dRdt]

# Initial conditions and parameters
initial_conditions = {
    HIGHLY_CAUTIOUS: {'S0': 800, 'I0': 0, 'R0': 0},
    MODERATELY_CAUTIOUS: {'S0': 150, 'I0': 0, 'R0': 0},
    LOW_CAUTIOUS: {'S0': 50, 'I0': 1, 'R0': 0}  # Start with one infected individual in low cautious group
}

# Transmission rates for each group
beta = [0.5, 1.0, 1.5]

# Recovery rate
gamma = 0.3

# Total populations
N = [initial_conditions[HIGHLY_CAUTIOUS]['S0'],
     initial_conditions[MODERATELY_CAUTIOUS]['S0'],
     initial_conditions[LOW_CAUTIOUS]['S0']]

# Initial conditions vector
y0 = [initial_conditions[HIGHLY_CAUTIOUS]['S0'], 
      initial_conditions[MODERATELY_CAUTIOUS]['S0'], 
      initial_conditions[LOW_CAUTIOUS]['S0'],
      initial_conditions[HIGHLY_CAUTIOUS]['I0'] \
     +initial_conditions[MODERATELY_CAUTIOUS]['I0'] \
     +initial_conditions[LOW_CAUTIOUS]['I0'], 0]  # SH, SM, SL, I, R

# Time grid (in days)
t = np.linspace(0, 100, 1600)

# Integrate the SIR equations over the time grid
ret = odeint(deriv, y0, t, args=(N, beta, gamma))

# Extract results
SH, SM, SL, I, R = ret.T

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t, SH, 'b', label='Highly Cautious Susceptible')
plt.plot(t, SM, 'g', label='Moderately Cautious Susceptible')
plt.plot(t, SL, 'r', label='Low Cautious Susceptible')
plt.plot(t, I, 'y', label='Infected')
plt.plot(t, R, 'k', label='Recovered')
plt.xlabel('Days')
plt.ylabel('Population')
plt.title('SIR Model with Different Caution Levels')
plt.legend()
plt.grid()
plt.show()

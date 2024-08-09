# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:18:26 2024

@author: joonc
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def deriv(y, t, beta, gamma):
    """
    Compute the derivatives for the SIR model with normalized population.

    Parameters:
    y : list
        A list containing the current values of s, i, r (proportions).
    t : float
        The current time point.
    beta : float
        The contact rate (parameter r in the problem).
    gamma : float
        The recovery rate (parameter a in the problem).

    Returns:
    dsdt, didt, drdt : float
        Derivatives of s, i, r (proportions).
    """
    s, i, r = y
    dsdt = -beta * s * i
    didt = beta * s * i - gamma * i
    drdt = gamma * i
    return dsdt, didt, drdt

# Initial conditions for part (a)
S0 = 762
I0 = 1
R0 = 0
N = S0 + I0 + R0

# Normalize initial conditions
s0 = S0 / N
i0 = I0 / N
r0 = R0 / N

# Parameters for part (a)
beta = 0.066  # parameter r
gamma = 0.044    # parameter a

# Time grid
t = np.linspace(0, 300, 3000)  # Adjust time scale as needed

# Initial conditions vector (normalized)
y0 = s0, i0, r0

# Solve ODE
ret = odeint(deriv, y0, t, args=(beta, gamma))
s, i, r = ret.T

# Plot results for part (a)
plt.figure(figsize=(10, 6))
plt.plot(t, s, label='Susceptible')
plt.plot(t, i, label='Infected')
plt.plot(t, r, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Proportion of Population')
plt.title('SIR Model (Normalized) - Part (a)')
plt.legend()
plt.grid(True)
plt.show()
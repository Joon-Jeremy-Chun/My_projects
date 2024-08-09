# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 14:10:42 2024

@author: joonc
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def deriv(y, t, N, beta, gamma):
    """
    Compute the derivatives for the SIR model.

    Parameters:
    y : list
        A list containing the current values of S, I, R.
    t : float
        The current time point.
    N : float
        The total population.
    beta : float
        The contact rate.
    gamma : float
        The recovery rate.

    Returns:
    dSdt, dIdt, dRdt : float
        Derivatives of S, I, R.
    """
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions
S0 = 8000
I0 = S0 * 0.0001
R0 = 0
N = S0 + I0 + R0

# Parameters
beta = 0.06
gamma = 0.001

# Time grid
t = np.linspace(0, 300, 3000)

# Initial conditions vector
y0 = S0, I0, R0

# Solve ODE
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.grid(True)
plt.show()

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 22:26:09 2024

@author: joonc
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# SIR model differential equations.
def deriv(y, t, N, beta, gamma, delta):
    S, I, R, D = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I - delta * I
    dRdt = gamma * I
    dDdt = delta * I
    return dSdt, dIdt, dRdt, dDdt

# Initial number in ratio of infected, recovered, death individuals, everyone else is susceptible to infection initially.
S0 = 999
I0 = 1
R0 = 0
D0 = 0

N = S0 + I0 + R0 + D0

# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta = 0.5
gamma = 0.2
delta = 0.02

# A grid of time points (in days)
t = np.linspace(0, 160, 160)

# Initial conditions vector
y0 = S0, I0, R0, D0

# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma, delta))
S, I, R, D = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.plot(t, D, label='Death')
plt.xlabel('Days')
plt.ylabel('Population')
plt.title('SIR Model Example')
plt.legend()
plt.show()
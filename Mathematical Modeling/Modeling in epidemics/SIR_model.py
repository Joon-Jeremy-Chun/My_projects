# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:53:23 2024

@author: joonc
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial number in ratio of infected and recovered individuals, everyone else is susceptible to infection initially.
I0 = 1/1000
R0 = 199/1000
S0 = 800/1000
N = S0 + I0 + R0

# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta = 0.5
gamma = 0.2

# A grid of time points (in days)
t = np.linspace(0, 160, 160)

# Initial conditions vector
y0 = S0, I0, R0

# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Population')
plt.title('SIR Model Example')
plt.legend()
plt.show()
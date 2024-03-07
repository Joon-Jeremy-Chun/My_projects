# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 20:50:17 2024

@author: joonc
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = mu*N -beta * S * I / N - mu*S
    dIdt = beta * S * I / N - (gamma + mu) * I
    dRdt = gamma * I - mu * R
    return dSdt, dIdt, dRdt

# Initial number in ratio of infected and recovered individuals, everyone else is susceptible to infection initially.
I0 = 1
R0 = 0
S0 = 1000
N = S0 + I0 + R0

# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta = 0.7
gamma = 0.5

# beta = 0.5
# gamma = 0.2
# mu = 0.1

# Birth and death rate are equal, mu
mu = 0.4 

# A grid of time points (in days)
t = np.linspace(0, 1600, 1600)

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

#Print Basic Reproduction Number R_0

R_0 = beta / (gamma + mu)
print("Basic Reproduction Number R_0:", R_0)
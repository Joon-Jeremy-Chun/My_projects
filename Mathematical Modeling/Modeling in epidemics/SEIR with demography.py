# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 21:12:23 2024

@author: joonc
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# SEIR model differential equations with demography.
def deriv(y, t, N, beta, gamma, sigma, mu):
    S, E, I, R = y
    dSdt = mu*N - beta * S * I / N - mu*S
    dEdt = beta * S * I / N - (sigma + mu) * E
    dIdt = sigma * E - (gamma + mu) * I
    dRdt = gamma * I - mu * R
    return dSdt, dEdt, dIdt, dRdt

# Initial number of exposed, infected, and recovered individuals,
# everyone else is susceptible to infection initially.
E0 = 0
I0 = 1
R0 = 0
S0 = 1000
N = S0 + E0 + I0 + R0

# Contact rate, beta, mean recovery rate, gamma, and incubation period, sigma (in 1/days).
beta = 0.5
gamma = 0.2
sigma = 0.1

# Birth and death rate are equal, mu
mu = 0.01

# A grid of time points (in days)
t = np.linspace(0, 1600, 1600)

# Initial conditions vector
y0 = S0, E0, I0, R0

# Integrate the SEIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma, sigma, mu))
S, E, I, R = ret.T

# Plot the data on four separate curves for S(t), E(t), I(t), and R(t)
plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Population')
plt.title('SEIR Model Example')
plt.legend()
plt.show()

# Print Basic Reproduction Number R_0
R_0 = beta / (gamma + mu)
print("Basic Reproduction Number R_0:", R_0)
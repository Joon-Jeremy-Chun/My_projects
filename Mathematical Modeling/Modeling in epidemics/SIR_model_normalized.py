# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 21:42:23 2024

@author: joonc
"""

#Normalized in S,I,R

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
#%%
# SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt
#%%
# Initial number in ratio of infected and recovered individuals, everyone else is susceptible to infection initially.
S0 = 762
#S0 = 800
I0 = 1
#I0 = S0*0.0001
R0 = 0

N = S0 + I0 + R0

S0 = S0 #Normalized
I0 = I0 #Normalized
R0 = R0 #Normalized

# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta = 2.18*(10**(-3))
gamma = 0.44
# beta = 1/2
# gamma = 1/3

# A grid of time points (in days)
t = np.linspace(0, 50, 3000)

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
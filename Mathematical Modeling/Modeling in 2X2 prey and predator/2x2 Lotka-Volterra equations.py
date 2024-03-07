# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 19:56:53 2024

@author: joonc
"""
# The Lotka-Volterra equations, also known as the predator-prey equations, are a pair of first-order, non-linear, differential equations
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the Lotka-Volterra equations
def lotka_volterra(t, y, alpha, beta, gamma, delta):
    prey, predator = y
    dydt = [alpha * prey - beta * prey * predator,
            delta * prey * predator - gamma * predator]
    return dydt

# Define the parameters
alpha = 0.1  # prey birth rate
beta = 0.02  # predation rate
gamma = 0.3  # predator death rate
delta = 0.01 # predator growth rate

# Initial conditions
prey_0 = 40
predator_0 = 9
y0 = [prey_0, predator_0]

# Time vector
t = np.linspace(0, 200, 1000)

# Solve the ODEs
sol = solve_ivp(lotka_volterra, [t[0], t[-1]], y0, args=(alpha, beta, gamma, delta), t_eval=t)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], 'b', label='Prey')
plt.plot(sol.t, sol.y[1], 'r', label='Predator')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Predator-Prey Model')
plt.legend()
plt.grid()
plt.show()
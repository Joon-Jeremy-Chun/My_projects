# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 22:30:51 2024

@author: joonc
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define parameters
beta = 0.3  # Transmission rate
gamma = 0.1  # Recovery rate
delta = 0.05  # Rate of vaccination

# Initial conditions
S0 = 0.8  # Initial susceptible population
I0 = 0.2  # Initial infectious population
V0 = 0.0  # Initial vaccinated population

# Time vector
t = np.linspace(0, 100, 1000)

# Define the SIS model equations with vaccination
def sis_model_vaccine(y, t, beta, gamma, delta):
    S, I, V = y
    
    # Rate of change equations
    dSdt = - beta * S * I + delta * V
    dIdt = beta * S * I - gamma * I
    dVdt = - delta * V + gamma * I
    
    return [dSdt, dIdt, dVdt]

# Initial conditions vector
y0 = [S0, I0, V0]

# Integrate the SIS model equations with vaccination over the time grid
solution = odeint(sis_model_vaccine, y0, t, args=(beta, gamma, delta))

# Extracting results
S, I, V = solution[:, 0], solution[:, 1], solution[:, 2]

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, I, 'r', label='Infectious')
plt.plot(t, V, 'g', label='Vaccinated')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIS Model with Vaccination')
plt.legend()
plt.grid()
plt.show()
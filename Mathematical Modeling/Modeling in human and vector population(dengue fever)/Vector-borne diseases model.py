# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 21:29:33 2024

@author: joonc
"""
#need work more on this code

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define parameters
beta_h = 0.1  # Transmission rate from human to mosquito
beta_m = 0.2  # Transmission rate from mosquito to human
gamma_h = 0.05  # Recovery rate of humans
gamma_m = 0.1  # Mortality rate of mosquitoes

# Initial conditions
S_h0 = 0.9  # Initial susceptible human population
I_h0 = 0.1  # Initial infectious human population
S_m0 = 0.8  # Initial susceptible mosquito population
I_m0 = 0.2  # Initial infectious mosquito population

# Time vector
t = np.linspace(0, 100, 1000)

# Define the SI model equations
def si_model(y, t, beta_h, beta_m, gamma_h, gamma_m):
    S_h, I_h, S_m, I_m = y
    
    # Rate of change equations for human population
    dS_hdt = - beta_h * S_h * I_m + gamma_h * I_h
    dI_hdt = beta_h * S_h * I_m - gamma_h * I_h
    
    # Rate of change equations for mosquito population
    dS_mdt = -beta_m * I_m * I_m * S_h  # No natural input for susceptible mosquitoes
    dI_mdt = beta_m * S_m * I_h - gamma_m * I_m
    
    return [dS_hdt, dI_hdt, dS_mdt, dI_mdt]

# Initial conditions vector
y0 = [S_h0, I_h0, S_m0, I_m0]

# Integrate the SI model equations over the time grid
solution = odeint(si_model, y0, t, args=(beta_h, beta_m, gamma_h, gamma_m))

# Extracting results
S_h, I_h, S_m, I_m = solution[:, 0], solution[:, 1], solution[:, 2], solution[:, 3]

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t, S_h, 'b', label='Susceptible Humans')
plt.plot(t, I_h, 'r', label='Infectious Humans')
plt.plot(t, S_m, 'g', label='Susceptible Mosquitoes')
plt.plot(t, I_m, 'y', label='Infectious Mosquitoes')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SI Model')
plt.legend()
plt.grid()
plt.show()
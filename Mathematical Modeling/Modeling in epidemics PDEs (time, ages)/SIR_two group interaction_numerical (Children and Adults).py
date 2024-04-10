# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:20:26 2024

@author: joonc
"""

import numpy as np
import matplotlib.pyplot as plt

# SIR model differential equations with two groups.
def deriv(y, Nc, Na, beta_cc, beta_ca, beta_ac, beta_aa, gamma_c, gamma_a, fg):
    Sc, Ic, Rc, Sa, Ia, Ra = y
    dScdt = -(beta_cc*Ic/Nc + beta_ac*Ia/Na)*Sc - fg*Sc 
    dIcdt = (beta_cc*Ic/Nc + beta_ac*Ia/Na)*Sc - gamma_c*Ic - fg*Ic
    dRcdt = gamma_c*Ic - fg*Rc
    dSadt = -(beta_aa*Ia/Na + beta_ca*Ic/Nc)*Sa + fg*Sc 
    dIadt = (beta_aa*Ia/Na + beta_ca*Ic/Nc)*Sa - gamma_a*Ia + fg*Ic 
    dRadt = gamma_a*Ia + fg*Rc 
    return np.array([dScdt, dIcdt, dRcdt, dSadt, dIadt, dRadt])

# Initial conditions
Ic0, Ia0, Rc0, Ra0, Sc0, Sa0 = 0, 1, 0, 0, 150, 850
y0 = np.array([Sc0, Ic0, Rc0, Sa0, Ia0, Ra0])

# Parameters
Nc = Sc0 + Ic0 + Rc0
Na = Sa0 + Ia0 + Ra0
beta_cc, beta_ca, gamma_c = 0.15, 0.3, 0.12
beta_aa, beta_ac, gamma_a = 0.35, 0.3, 0.1
fg = 0.001

# Time grid
t_end = 400
dt = 1  # Time step
N = int(t_end/dt)
t = np.linspace(0, t_end, N)

# Solution array
solution = np.zeros((N, 6))
solution[0] = y0

# Finite Difference Method
for i in range(1, N):
    y = solution[i-1]
    dydt = deriv(y, Nc, Na, beta_cc, beta_ca, beta_ac, beta_aa, gamma_c, gamma_a, fg)
    solution[i] = y + dydt * dt

# Extracting individual components
Sc, Ic, Rc, Sa, Ia, Ra = solution.T

# Plotting
plt.plot(t, Sc, label='Susceptible Children')
plt.plot(t, Ic, label='Infected Children')
plt.plot(t, Rc, label='Recovered Children')
plt.plot(t, Sa, label='Susceptible Adults')
plt.plot(t, Ia, label='Infected Adults')
plt.plot(t, Ra, label='Recovered Adults')
plt.xlabel('Days')
plt.ylabel('Population')
plt.title('SIR Model with Two Groups (Children and Adults) using FDM')
plt.legend()
plt.show()

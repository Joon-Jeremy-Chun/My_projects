# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:38:44 2024

@author: joonc
"""
#finding equilibrium

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Parameters
beta = 0.2   # Infection rate
gamma = 0.1  # Recovery rate

# Equations for the SIR model
def f_S(S, I, R, t):
    return -beta * S * I

def f_I(S, I, R, t):
    return beta * S * I - gamma * I

def f_R(S, I, R, t):
    return gamma * I

# Trapezoidal predictor-corrector method
def trapezoidal_pc(y0, t0, fS, fI, fR, h, N):
    t = np.zeros(N+1)
    S = np.zeros(N+1)
    I = np.zeros(N+1)
    R = np.zeros(N+1)
    S[0], I[0], R[0] = y0
    t[0] = t0

    for n in range(N):
        # Predictor Step
        S_pred = S[n] + h * fS(S[n], I[n], R[n], t[n])
        I_pred = I[n] + h * fI(S[n], I[n], R[n], t[n])
        R_pred = R[n] + h * fR(S[n], I[n], R[n], t[n])

        # Corrector Step
        S[n+1] = S[n] + h * (fS(S[n], I[n], R[n], t[n]) + fS(S_pred, I_pred, R_pred, t[n+1])) / 2
        I[n+1] = I[n] + h * (fI(S[n], I[n], R[n], t[n]) + fI(S_pred, I_pred, R_pred, t[n+1])) / 2
        R[n+1] = R[n] + h * (fR(S[n], I[n], R[n], t[n]) + fR(S_pred, I_pred, R_pred, t[n+1])) / 2
        t[n+1] = t[n] + h

    return t, S, I, R

# Find equilibrium points
def find_equilibrium():
    def equations(y):
        S, I, R = y
        eq1 = f_S(S, I, R, 0)
        eq2 = f_I(S, I, R, 0)
        eq3 = f_R(S, I, R, 0)
        return [eq1, eq2, eq3]

    # Initial guess
    y_guess = [0.9, 0.1, 0]

    # Solve equations
    S_eq, I_eq, R_eq = fsolve(equations, y_guess)

    return S_eq, I_eq, R_eq

# Initial conditions
t0 = 0
h = 0.1
N = 5000
S0 = 0.99
I0 = 0.01
R0 = 0
y0 = [S0, I0, R0]

# Simulate using Trapezoidal Predictor-Corrector method
t_max = 1000  # Maximum time for simulation
N = int(t_max / h)  # Number of time steps
t, S, I, R = trapezoidal_pc(y0, t0, f_S, f_I, f_R, h, N)

# Plot the results
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infectious')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model - Long-term Behavior')
plt.legend()
plt.grid(True)
plt.show()

# Extract long-term values
S_long_term = S[-1]
I_long_term = I[-1]
R_long_term = R[-1]

print("Long-term values:")
print("Susceptible (S) =", S_long_term)
print("Infectious (I) =", I_long_term)
print("Recovered (R) =", R_long_term)
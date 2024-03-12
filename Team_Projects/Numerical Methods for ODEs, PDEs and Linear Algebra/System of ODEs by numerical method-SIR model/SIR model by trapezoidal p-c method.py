# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 18:57:25 2024

@author: joonc
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
# Parameters
beta = 0.3   # Infection rate
gamma = 0.1  # Recovery rate
#%%
# ODE function
def f_S(S, I, R, t):
    dSdt = -beta*S*I
    return dSdt

def f_I(S, I, R, t):
    dIdt = beta*S*I - gamma*I
    return dIdt

def f_R(S, I, R, t):
    dRdt = gamma*I
    return dRdt

#%%
# Trapezoidal predictor-corrector method
# h:length of segement, N total iteration or Ns intervals
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

# Initial conditions
t0 = 0
h = 0.01
N = 10000
S0 = 0.999
I0 = 0.001
R0 = 0
y0 = [S0, I0, R0]

# Solve using Trapezoidal Predictor-Corrector method
t, S, I, R = trapezoidal_pc(y0, t0, f_S, f_I, f_R, h, N)

# Plot the results
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infectious')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Pupluation')
plt.title('SIR using Trapezoidal Predictor-Corrector Method')
plt.legend()
plt.grid(True)
plt.show()
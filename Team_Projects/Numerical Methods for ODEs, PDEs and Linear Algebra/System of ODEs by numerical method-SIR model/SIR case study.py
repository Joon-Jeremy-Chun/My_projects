# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:21:01 2024

@author: joonc
"""
# an introduction to numerical methods and analysis 2nd ed. by j. epperson.
# 6.8 Q11.

import numpy as np
import matplotlib.pyplot as plt

#%%
# Parameters
# r = 2.18 * 10**(-3)   # Infection rate
# a = 0.44  # Recovery rate
r=0.2
a =0.1
#%%
# ODE function
def f_S(S, I, R, t):
    dSdt = -r*S*I
    return dSdt

def f_I(S, I, R, t):
    dIdt = r*S*I - a*I
    return dIdt

def f_R(S, I, R, t):
    dRdt = a*I
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
#%%
# Initial conditions
t0 = 0
h = 0.1
N = 2000
S = 762
I = 1
R = 0

#Ratio
total_p = S + I + R
S0 = S / total_p
I0 = I / total_p
R0 = R / total_p
y0 = [S0, I0, R0]

# Solve using Trapezoidal Predictor-Corrector method
t, S, I, R = trapezoidal_pc(y0, t0, f_S, f_I, f_R, h, N)
#%%
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

R1 = np.zeros(N+1)
print(R1)
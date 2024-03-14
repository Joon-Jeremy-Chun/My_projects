# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:51:54 2024

@author: joonc
"""

# an introduction to numerical methods and analysis 2nd ed. by j. epperson.
# 6.8 Q11.
# Now consider the case where S0 = 10^6, R0 = 0, a = 1.74, r = 0.3 x 10^(-5). 
# Solve the system (again, using the method of your choice) for a range of values 
# of I0 > 0. Is there a critical value of I0 beyond which the disease spreads to 
# almost all of the population? These data are taken, roughly, from a study of a 
# plague epidemic in Bombay; the time scale is weeks. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%%
# Parameters
r = 0.3 * 10**(-5)   # Infection rate
a = 1.74  # Recovery rate

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
# h: length of a segment, N total iteration or Ns intervals
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
N = 200

S0 = 10**6
I0 = 1778279.410038923
R0 = 0
y0 = [S0, I0, R0]

# Solve using Trapezoidal Predictor-Corrector method
t, S, I, R = trapezoidal_pc(y0, t0, f_S, f_I, f_R, h, N)
#%%
#  1.How long does the epidemic run its course? .............
# We assume that the epidemic has ended when the number of infected people drops below 1.

# Find the time when the number of infected people drops below 1
end_time = 0
for i in range(len(I)):
    if I[i] < 1:
        end_time = i * h
        break

print("The epidemic runs its course until week:", end_time)    
#%%
#  2.When does it reach the peak? ............

Max_peak = max(I)
Max_week = pd.Series(I).idxmax() * h
print("The day of reachs to the peak: ", Max_week)
print("The number infected pepole: ", Max_peak)

#%%
# 3.Extract long-term values
# How many percent of the population is infected by the disease?
S_long_term = S[-1]
I_long_term = I[-1]
R_long_term = R[-1]

print("Long-term values:")
print("Susceptible (S) =", S_long_term)
print("Infectious (I) =", I_long_term)
print("Recovered (R) =", R_long_term)
total_p = S[0]+I[0]+R[0]
total_infected = R[-1]/total_p
print("Percent of infected =", total_infected*100)
#%%
#4.Plot the results
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infectious')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Pupluation')
plt.title('Initial infected: 10^5')
plt.legend()
plt.grid(True)
plt.show()

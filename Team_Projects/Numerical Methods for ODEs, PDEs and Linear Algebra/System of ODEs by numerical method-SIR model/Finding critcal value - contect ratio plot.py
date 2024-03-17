# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 22:03:30 2024

@author: Matthew Perez
"""

from matplotlib import pyplot as plt
import numpy as np
import math

# r and a  values
# original r = .3 * (10**(-5)) = .000003

r = .000001725 # This is our r critical value
#r = .000001725 + .000001 # This is our r value that will lead to a peak and then rise
#r = .000001725 - .000001 # This is our r value that will lead to a monotonically decreasing infection curve
a = 1.74
h = .1

# Defining our S differential equation
def dS(S, I, R, t):
    value = -r * S * I
    return value

# Defining our I differential equation
def dI(S, I, R, t):
    value = r * S * I - (a * I)
    return value

# Defining our R differential equation
def dR(S, I, R, t):
    value = a * I
    return value

# Here are our initial conditions
S_0 = 10**6
I_0 = 10**4
R_0 = 0
# We use N to create the length of our vectors and time interval
N = 1001
# Vectors to store approximations to SIR Model
S = np.zeros(N)
I = np.zeros(N)
R = np.zeros(N)
T = np.zeros(N)

# Trapezoidal Corrector Method
for j in range(1, len(T) + 1):
    if j == 1:
        # This sets our initial conditions as the first element in our S, I, R, T vectors
       S[j-1] = S_0
       I[j-1] = I_0
       R[j-1] = R_0
       T[j-1] = 0
    else:
        # Euler Predictor Method
        S_temp = S[j - 2] + (h * dS(S[j - 2], I[j - 2], R[j - 2], T[j - 2]))
        I_temp = I[j - 2] + (h * dI(S[j - 2], I[j - 2], R[j - 2], T[j - 2]))
        R_temp = R[j - 2] + (h * dR(S[j - 2], I[j - 2], R[j - 2], T[j - 2]))
        T_temp = T[j - 2]

        # Trapezoidal Corrector
        S[j-1] = S[j-2] + ((h/2) * (dS(S[j-2], I[j-2], R[j-2], T[j-2]) + dS(S_temp, I_temp, R_temp, T_temp)))
        I[j-1] = I[j-2] + ((h/2) * (dI(S[j-2], I[j-2], R[j-2], T[j-2]) + dI(S_temp, I_temp, R_temp, T_temp)))
        R[j-1] = R[j-2] + ((h/2) * (dR(S[j-2], I[j-2], R[j-2], T[j-2]) + dR(S_temp, I_temp, R_temp, T_temp)))
        T[j-1] = T_temp + h # This line creates the respective t_j time index for the jth element in our loop


# Plotting all of our solution curves
plt.plot(T, S, label = 'Susceptible')
plt.plot(T, I, label = 'Infected')
plt.plot(T, R, label = 'Recovered')
plt.title(f'Solution Curves to SIR Model, r = {r}')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()

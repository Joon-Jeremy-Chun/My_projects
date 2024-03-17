# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 21:51:07 2024

@author: Matthew Perez
"""
from matplotlib import pyplot as plt
import numpy as np
import math

# I_0 = 10**4, vary r, Our goal is to find:
# 1) r < r_crit so number of infected drops monotonically
# 2) r > r_crit so number rises to a peak and then drops off
# We need to use trapezoidal corrector method with h = .01

# r and a  values
# original r = .3 * (10**(-5))
a = 1.74
h = .1

# Defining our S differential equation
def dS(S, I, R, t, r):
    value = -r * S * I
    return value

# Defining our I differential equation
def dI(S, I, R, t, r):
    value = r * S * I - (a * I)
    return value

# Defining our R differential equation
def dR(S, I, R, t, r):
    value = a * I
    return value

# Initial conditions
S_0 = 10**6
I_0 = 10**4
R_0 = 0
# Used to create "length" of time interval
N = 1001
# Vectors to store solution values to dS, dI, dR, time vector
S = np.zeros(N)
I = np.zeros(N)
R = np.zeros(N)
T = np.zeros(N)

# Vector to store r values to test for critical points
r_values = [0]
# Vector to store r values that cause infection curve to monotonically decrease
r_reduce =[]
# Vector to store r values that cause infection curve to increase
r_increase =[]

# This is a for loop to create the r values we will be testing out in this problem
for j in range(1, 61):
    value = .005*j
    value_upd = value * (10**(-5))
    r_values.append(value_upd) # stores test r values into r_values vector

for Q in r_values:
    # Trapezoidal Corrector Method
    # We only need to test for the second time index and see if the value decreases, or increases compared to I_0
    for j in range(1, 3):
        if j == 1:
            # This sets our initial conditions as the first element in our S, I, R, T vectors
            S[j - 1] = S_0
            I[j - 1] = I_0
            R[j - 1] = R_0
            T[j - 1] = 0
        else:
            # Euler Predictor Method
            S_temp = S[j - 2] + (h * dS(S[j - 2], I[j - 2], R[j - 2], T[j - 2], Q))
            I_temp = I[j - 2] + (h * dI(S[j - 2], I[j - 2], R[j - 2], T[j - 2], Q))
            R_temp = R[j - 2] + (h * dR(S[j - 2], I[j - 2], R[j - 2], T[j - 2], Q))
            T_temp = T[j - 2]

            # Trapezoidal Corrector
            S_Test = S[j-2] + ((h/2) * (dS(S[j-2], I[j-2], R[j-2], T[j-2], Q) + dS(S_temp, I_temp, R_temp, T_temp, Q)))
            I_Test = I[j-2] + ((h/2) * (dI(S[j-2], I[j-2], R[j-2], T[j-2], Q) + dI(S_temp, I_temp, R_temp, T_temp, Q)))
            R_Test = R[j-2] + ((h/2) * (dR(S[j-2], I[j-2], R[j-2], T[j-2], Q) + dR(S_temp, I_temp, R_temp, T_temp, Q)))
            T_Test = T_temp + h # This line creates the respective t_j time index for the jth element in our loop

            if I_Test < I[0]:
                r_reduce.append(Q)
                # store value in r_reduce if second I_1 < I_0
            elif I_Test > I[0]:
                r_increase.append(Q)
                # store value in r_increase if I_1 > I_0
            else:
                print(f"{Q} does not alter infections curve after 1 immediate time indices t.")
                # Put this here to see if one of our test r values would get I_1 = I_0, but it isn't guaranteed

print(r_reduce)
print('')
print(r_increase)
print('')

# This averages the last test r entry stored in r_reduce and the first test r entry stored in r_increase
# We are essentially averaging r_1 and r_2, where:
# r_1 causes infection solution curve to monotonically decrease, but this is the last test value r to do this
# r_2 causes infection solution curve to rise to a peak and then fall, but this is the first test value r to do this
r_crit = (r_reduce[-1] + r_increase[0])/2

# r_crit +.000001
r_critplus = r_crit + .000001

# r_crit - .000001
r_critminus = r_crit - .000001


print(f"Our r critical value is approximately {r_crit}")
print(f"r_crit + .000001 = {r_critplus}")
print(f"r_crit - .000001 = {r_critminus}")


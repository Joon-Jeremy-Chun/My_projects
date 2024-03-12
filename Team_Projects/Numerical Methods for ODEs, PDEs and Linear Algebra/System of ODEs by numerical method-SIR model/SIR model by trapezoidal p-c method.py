# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 18:57:25 2024

@author: joonc
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
# ODE function
def f_c(S, I, R, t):
    a = 3
    return y*(a-y)

#%%
# Trapezoidal predictor-corrector method
# h:length of segement, N total iteration or Ns intervals
def trapezoidal_pc(y0, t0, f, h, N):
    t = np.zeros(N+1)
    y = np.zeros(N+1)
    y[0] = y0
    t[0] = t0

    for n in range(N):
        # Predictor Step
        y_pred = y[n] + h * f(y[n], t[n])

        # Corrector Step
        y[n+1] = y[n] + h * (f(y[n], t[n]) + f(y_pred, t[n+1])) / 2
        t[n+1] = t[n] + h

    return t, y

# Initial conditions
y0 = 0.001
t0 = 0
h = 0.01
N = 1000

# Solve using Trapezoidal Predictor-Corrector method
t, y = trapezoidal_pc(y0, t0, h, N)

# Plot the results
plt.plot(t, y, label='Trapezoidal Predictor-Corrector')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Solution of ODE using Trapezoidal Predictor-Corrector Method')
plt.legend()
plt.grid(True)
plt.show()
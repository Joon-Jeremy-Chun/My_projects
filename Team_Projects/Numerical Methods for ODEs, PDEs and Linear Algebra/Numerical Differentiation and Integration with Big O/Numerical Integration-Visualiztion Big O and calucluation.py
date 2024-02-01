# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 17:26:38 2024

@author: joonc
"""

#The data from previous
import numpy as np
from scipy import integrate
import math
import d1

#%%
def T_Sin(x):
        return np.sin(x)
#%%
def integrand(x):
    return np.sin(x)

# Specify the integration limits
lower_limit = 0
upper_limit = np.pi*2

# Use the quad function for integration
exact_v, error = integrate.quad(integrand, lower_limit, upper_limit)
#%%
a = 0
b = np.pi*2
t = 20 # max N

errors = []
for i in range(1,t):
    S = d1.SimC(T_Sin, a, b, 2**i + 1)# need odd nodes
    Err_S = abs(exact_v - S)
    errors.append(Err_S)
#%%
print(errors)
log_values = [math.log(value) for value in errors]

# Print the result
print(log_values)

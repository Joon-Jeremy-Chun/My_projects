# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 17:26:38 2024

@author: joonc
"""

#The data from previous
import matplotlib.pyplot as plt
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
#Define the N values
t = 20
N_tri = []
for i in range(1,t):
    N_tri.append(2**i + 1)
print(N_tri)
#%%
#calculate errors by each N values
a = 0
b = np.pi*2
t = 20 # max N

errors = []
for i in range(1,t):
    S = d1.SimC(T_Sin, a, b, 2**i + 1)# need odd nodes
    Err_S = abs(exact_v - S)
    errors.append(Err_S)
#%%
log_values_e = [math.log(value) for value in errors]

# Print the result
print(log_values_e)
#%%
log_value_n = [math.log(k) for k in N_tri]
print(log_value_n)
#%%
x = [0,1,2,3,4,5,6,7,8,9,10,11,12]
y = [0,-2,-4,-6,-8,-10,-12,-14,-16,-18,-20,-22,-24]
plt.plot(log_value_n, log_values_e, label='error in log')
plt.plot(x,y, label='test line slope 2' )
plt.title('Graph')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
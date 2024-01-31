# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 09:38:07 2024

@author: joonc
"""

#Objectives:Finding Big O for Simpson's rule by given functions

import numpy as np
from scipy import integrate
#%%
#F : functions, a: starting point, b: Ending point, N: # of segments 
def SimC(F, a, b, N):
    Sum = 0  
    h = (b - a) / N
    
    #Simpson's Rule need odd terms, and a > b
    if a >= b:
        print("cannot apply the inegration; a >= b")
        return None
    
    if N%2 == 0:
        print("cannot apply the Simpon's Rule; even # of terms")
        return None
        
    else:
        for i in range(N):
            if i%2 != 0:
                Sum += 4*F(a + i*h)
            else:
                Sum += 2*F(a + i*h)
        #the First and Last terms correction
        Sum += -F(a)
        Sum += -F(b)
        
        return h/3*Sum
    
#Test Function
def T_Sin(x):
        return np.sin(x)
#%%
#Test Function with very Ns
a = 0
b = np.pi*2
N = 101
for N in range(1, N, 2): #range must be odd #
    result = SimC(T_Sin, a, b, N)
    print(f"For N = {N}, Result = {result}")
#%%    
#Finding Big O. 
#Comput the absolute error for all each terms
#Compare the errors changes
#%% 
import numpy as np
from scipy import integrate

# Define the function to be integrated
def integrand(x):
    return np.sin(x)

# Specify the integration limits
lower_limit = 0
upper_limit = np.pi

# Use the quad function for integration
result, error = integrate.quad(integrand, lower_limit, upper_limit)

# Print the result and error
print(f"Result of integration: {result}")
print(f"Estimated error: {error}")

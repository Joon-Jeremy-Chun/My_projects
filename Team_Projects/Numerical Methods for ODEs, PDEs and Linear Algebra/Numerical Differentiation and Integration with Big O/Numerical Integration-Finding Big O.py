# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 09:38:07 2024

@author: joonc
"""

#Objectives:Finding Big O for Simpson's rule by given functions

import numpy as np
from scipy import integrate
#%%
#F : functions, a: starting point, b: Ending point, N: # of Segments
def SimC(F, a, b, N):
    Sum = 0  
    h = (b - a) / N
    
    #Simpson's Rule needs odd terms, and a > b
    if a >= b:
        print("cannot apply the integration; a >= b")
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
# #Test Function with very Ns
# a = 0
# b = np.pi*2
# N = 101
# for N in range(1, N, 2): #range must be odd #
#     S = SimC(T_Sin, a, b, N)
#     print(f"For N = {N}, Result = {S}")
#%%    
#Finding Big O. 
#Comput the absolute error for all each terms
#Compare the errors changes
#%% 
# integrate exact value
def integrand(x):
    return np.sin(x)

# Specify the integration limits
lower_limit = 0
upper_limit = np.pi*2

# Use the quad function for integration
exact_v, error = integrate.quad(integrand, lower_limit, upper_limit)

# # Print the result and error
# print(f"Result of integration: {exact_v}")
# print(f"Estimated error: {error}")
#%%
#comput the absolute errers
#Trial 1
#increase N value with 2^t + 1, t = 1,2,...
#check the abs value differe from nth term to nth -1 terms
a = 0
b = np.pi*2
t = 20 # max N

errors = []
for i in range(1,t):
    S = SimC(T_Sin, a, b, 2**i + 1)# need odd nodes
    #print(f"For N = {i}, Result = {S}")
    Err_S = abs(exact_v - S)
    errors.append(Err_S)
    print(f"For N = 10**{i}+1, abs_err = {Err_S}")
#%%
#comput the absolute errers
#Trial 2
#increase N value with 10^t + 1, t = 1,2,...
#check the abs value differe from nth term to nth -1 terms
a = 0
b = np.pi*2
t = 8 # max N

errors = []
for i in range(1,t):
    S = SimC(T_Sin, a, b, 10**i + 1)# need odd nodes
    #print(f"For N = {i}, Result = {S}")
    Err_S = abs(exact_v - S)
    errors.append(Err_S)
    print(f"For N = 10**{i}+1, abs_err = {Err_S}")
#%%
#Check the Big O
#Since we     
error_ratios = [errors[i] / errors[i-1] for i in range(1, len(errors))]
print(error_ratios)
for i in range(1, len(errors)):
    rrors[i] / errors[i-1]
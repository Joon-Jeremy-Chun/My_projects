# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 21:57:19 2024

@author: joonc
"""

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
#%%
import numpy as np
#Test Function
def T_Sin(x):
        return np.sin(x)
#%%
a = 0
b = np.pi
t = 20 # max N

errors = []
for i in range(1,t):
    S = SimC(T_Sin, a, b, 2**i + 1)# need odd nodes
    print(f"For N = 2**{i}+1, approx = {S}")
    print(S - 2)
    print(abs(S-2))
    errors.append(S)
    
#%%
a = 0
b = np.pi
t = 20 # max N
exact_v = 2
abs_errors = []
for i in range(1,t):
    S = SimC(T_Sin, a, b, 2**i + 1)# need odd nodes
    Err_S = abs(exact_v - S)
    abs_errors.append(Err_S)
    print(f"For N = 10**{i}+1, abs_err = {Err_S}")
#%%
error_ratios = [abs_errors[i] / abs_errors[i-1] for i in range(1, len(abs_errors))]
print(error_ratios)
#%%
log_values = [math.log(value) for value in abs_errors]

# Print the result
print(log_values)

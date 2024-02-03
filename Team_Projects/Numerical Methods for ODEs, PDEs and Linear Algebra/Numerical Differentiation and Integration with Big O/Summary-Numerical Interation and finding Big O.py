# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 21:57:19 2024

@author: joonc
"""
# Summary-Numerical Interation and finding Big O
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import integrate

#%%
#Simpson's rules
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
#Test Function
def T_Sin(x):
        return (1-x**2)**(1/2)
a = -1
b = 1
t = 20 # max N
#%%
#compute the exact value of Test function
#%% 
# integrate exact value
# Use the quad function for integration
exact_v, error_exact_v = integrate.quad(T_Sin, a, b)
#%%
#Define the N values or h sequences
N_tri = []
for i in range(1,t):
    N_tri.append(2**i + 1)
print(N_tri)
#%%
#compute the error
aprox = []
for i in N_tri:
    S = SimC(T_Sin, a, b, i)# need odd nodes
    print(f"For N = {i}, approx = {S}")
    aprox.append(S)
    
#%%
#compute the absolute error
abs_errors = []
for i in aprox:
    Err_S = abs(exact_v - i)
    abs_errors.append(Err_S)
    print(f"For N = {i}, abs_err = {Err_S}")
#%%
#finding the pattern of the absolute errors
error_ratios = [abs_errors[i] / abs_errors[i-1] for i in range(1, len(abs_errors))]
print(error_ratios)

#%%
#Covert into the natureal loge scales
log_values_e = [math.log(e) for e in abs_errors]
log_value_n = [math.log(k) for k in N_tri]
#%%
x = [0,1,2,3,4,5,6,7,8,9,10,11,12]
y = [0,-2,-4,-6,-8,-10,-12,-14,-16,-18,-20,-22,-24]
plt.plot(log_value_n, log_values_e, label='abs_error in log')
plt.plot(x,y, label='test line: slope 2' )
plt.title('The patterns of abs_error')
plt.xlabel('N_tri (ln)')
plt.ylabel('abs_error (ln)')
plt.legend()
plt.grid(True)
plt.show()

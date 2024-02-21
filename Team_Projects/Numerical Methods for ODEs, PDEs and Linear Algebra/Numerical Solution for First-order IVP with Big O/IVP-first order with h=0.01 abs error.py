# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:46:07 2024

@author: joonc
"""

# Implement the Midpoint method with h = 0.01 to approximate the solution to the above IVP and compare it with the exact solution.

import numpy as np
#%%
#Define inline function
def model(t,y):
    dydt = (2/t)*y +(t**2)*np.exp(t)
    return dydt

#%%
#Define midpoint method
def Mid_ptn(f, y0, t):
    h = (b-a) / (len(t)-1)
    y =[]
    y.append(y0)
    y1 = y0+h*model(t[0],y0)
    y.append(y1)
    
    for i in t[1:-1]: 
        yn = y[-2] + 2*h*f(i,y[-1])
        y.append(yn)
    return y
#%%
#Define initial condition
y0 = 0
#Define time and invervals
N= 100 #intervals # h=0.01
a = 1 #starting
b = 2 #ending
t = np.linspace(a,b,N+1) #N+1 points
#%%
y = Mid_ptn(model, y0, t)
#%%
#Define exact function y and exact value at b
def Exact_f(t):
    y = t**2*(np.exp(t)-np.exp(1))
    return y
#%%
#compute approximate value, exact value, and absoulte error
Exact_y = Exact_f(b) #exact value
apx = y[-1] #approximate
abs_error = abs(y[-1] - Exact_y) #absolute error
print('excat_y: ', Exact_y)
print('approximate: ', apx )
print('absolute error: ', abs_error)

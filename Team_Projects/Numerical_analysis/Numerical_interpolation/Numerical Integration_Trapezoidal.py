# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 09:55:04 2023

@author: joonc
"""
#Numertical integration;Trapezoid rule by given data points or functions
#1. Use the given data points
#%%
#Define the Trapezid integration by given points
def Ta(Y, h):
    Sum = 0
    
    for i in range(len(Y)):
        Sum += 2*Y[i]
        
    #the first and last terms correction
    Sum += -Y[0]
    Sum += -Y[-1]   
        
    return h/2*Sum
#%%
#The result of Tapezoidal intergration
X = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84]
Y = [124, 134, 148, 156, 147, 133, 121, 109, 99, 85, 78, 89, 104, 116, 123]

print('-trapezoidal-')
print('In feets:', Ta(Y, 6))
print('In miles:', Ta(Y,6)*0.000189394)
    
#%%

import numpy as np

#2. Use the given functions
#Given function
def f(x):
    return np.sin(x)

#%%
#define Trapezoid intergration

#f;function, n; # of intervals, a;left end, b; right end
def Trf(f, n, a, b):
    area = 0
    
    for i in range(n):
        
        h = abs(b-a)/n
        x_i = i(b-a)/n
        

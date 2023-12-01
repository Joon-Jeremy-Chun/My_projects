# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 12:10:57 2023

@author: joonc
"""
#Numertical integration;Trapezoid rule by given functions
#2. Use the given functions
#%%

import numpy as np
#%%
#define Trapezoid intergration

#f;function, n; # of intervals, a;left end, b; right end
def Trf(f, n, a, b):
    area = 0
    
    for i in range(n):
        
        h = abs(b-a)/n
        x_i = i(b-a)/n + a
        area_i = 2*f(x_i)
        
        area += area_i
        
    #correction, the first and the last terms
    area += -f(a)
    area += -f(b)
        
    return h/2*area
#%%
#Ex1)
#Given function
def f(x):
    return np.sin(x)

#%%
#Ex1)
#Given function
def g(x):
    return x**2

#%%
print(Trf(g, 1000, 1,10))
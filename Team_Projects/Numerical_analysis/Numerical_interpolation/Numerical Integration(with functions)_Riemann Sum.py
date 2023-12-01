# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 12:01:46 2023

@author: joonc
"""
#Numertical integration;left Riemann sum by given functions
#2. Integrate by the given functions
import numpy as np

#%%
# Define the function
def f(x):
    return np.sin(x)

#%%
# Define the function
def g(x):
    return x**2

#%%
# Polynomial funciton from Brad's works

def b(x):
    return 118.515*x**0 + 5.276*x**1 - 0.244*x**2 + 0.003*x**3 + -1.175*x**4


#%%
# Define left Riemann sum
def lrs( f, n, a, b):
    #f;function, n; # of intervals, a;left end, b; right end
    area = 0
    for i in range(n):
        
        h = abs(b-a)/n #define h
        x_i = i*abs(b-a)/n #dfine x_i value
        area_i = f(x_i)*h #ith rectangle area = f(x_i) * h
        #print(area_i)
        area += area_i
        
    return area
#%%
#test
print(lrs(f, 3, 0, np.pi))
print(lrs(f, 1000, 0, np.pi))
#%%
#test2
print(lrs(g, 3, 0, 10))
print(lrs(g, 1000, 0, 10))
print(lrs(g, 10000, 0, 10))

#%%
print(lrs(b, 15, 0, 90))
print(lrs(b, 1000, 0, 90))


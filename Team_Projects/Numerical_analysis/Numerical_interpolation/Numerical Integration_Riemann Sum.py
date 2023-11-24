# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 12:34:26 2023

@author: joonc
"""
#Numertical integration;left Riemann sum by given data points or functions

#1. Integration by given data points
import pandas as pd

#%%

# Given data points x = time (s) y = speed (feet/second)
X = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84]
Y = [124, 134, 148, 156, 147, 133, 121, 109, 99, 85, 78, 89, 104, 116, 123]

data = {
        'time' : X,
        'speed' : Y 
        }

df = pd.DataFrame(data)
#%%
#At first, to get the idea approximate for the length of the track in mile. Use left Riemann sum 
#length of the track (feet) = speed (feet/second) * time_interval (second)

Sf = 0
for i in range(len(df)):
    a = 6
    b = df.iloc[i]['speed']
    c = a*b
    Sf += c

#left sied Riemann sum
print('-left Riemann sum-')
print('In feets:',Sf,)

#change the units: feet -> miles
# 1 feet = 0.000189394 miles
Sm = Sf*0.000189394
print('In miles:',Sm,'miles')

#%%
#2. Integrate by the given functions
import numpy as np


#%%
# Define the function
def f(x):
    return np.sin(x)

#%%
# Define left riemann sum
def lrs( f, n, a, b):
    #f;function, n; # of intervals, a;left end, b; right end
    area = 0
    for i in range(n):
        
        h = abs(b-a)/n #define h
        x_i = i*(b-a)/n #dfine x_i value
        area_i = f(x_i)*h #ith rectangle area = f(x_i) * h
        #print(area_i)
        area += area_i
        
    return area
#%%
#test
print(lrs(f, 3, 0, np.pi))
print(lrs(f, 1000, 0, np.pi))


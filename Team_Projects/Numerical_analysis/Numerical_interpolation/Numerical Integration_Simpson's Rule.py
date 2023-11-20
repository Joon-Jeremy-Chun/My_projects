# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 12:18:59 2023

@author: joonc
"""

#%%
#Use the given data points
#Choose the numerical integration method - Simpson's Rule

def Sim(Y, h):
    Sum = 0
        
    for i in range(len(Y)):
        if i%2 != 0: #odd terms
            Sum += 4*Y[i]
        else: #even terms
            Sum += 2*Y[i]
            
    #the first and last terms correction
    Sum += -Y[0]
    Sum += -Y[-1]    
    
    return h/3*Sum
#%%
#Calculate the Simpson's Rule

X = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84]
Y = [124, 134, 148, 156, 147, 133, 121, 109, 99, 85, 78, 89, 104, 116, 123]

print("-Simpson's Rule-")
print('In feet:', Sim(Y, 6))
print('In mile:', Sim(Y, 6)*0.000189394)
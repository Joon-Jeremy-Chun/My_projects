# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 09:55:04 2023

@author: joonc
"""
#Numertical integration;Trapezoid rule (for data points)
#1. Use the given data points
#%%
#Define the Trapezid integration (for data points)
def Ta(Y, h):
    Sum = 0
    
    for i in range(len(Y)):
        Sum += 2*Y[i]
        
    #the first and last terms correction
    Sum += -Y[0]
    Sum += -Y[-1]   
        
    return h/2*Sum
#%%
#length of the track (feet) = speed (feet/second) * time_interval (second)
#The result of Tapezoidal intergration
X = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84]
Y = [124, 134, 148, 156, 147, 133, 121, 109, 99, 85, 78, 89, 104, 116, 123]

print('-Trapezoidal-')
print('In feets:', Ta(Y, 6))
print('In miles:', Ta(Y,6)*0.000189394)        

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 09:55:04 2023

@author: joonc
"""

#Numerical Integration

import pandas as pd

#%%
# Given data x = time (s) y = speed (feet/second)
X = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84]
Y = [124, 134, 148, 156, 147, 133, 121, 109, 99, 85, 78, 89, 104, 116, 123]

data = {
        'time' : X,
        'speed' : Y 
        }

df = pd.DataFrame(data)
#%%
#At first, to get the idea approximate for the length of the track in mile. Use left riemann sum 
#length of the track (feet) = speed (feet/second) * time_interval (second)

Sf = 0
for i in range(len(df)):
    a = 6
    b = df.iloc[i]['speed']
    c = a*b
    Sf += c

print('nearly:',Sf,'feet')

#change the units: feet -> miles
# 1 feet = 0.000189394 miles
Sm = Sf*0.000189394
print('nearly:',Sm,'miles')
#%%
#Use the given data points
#Choose the numerical integration method - Trapezoidal Rule

def Ta(Y, h):
    Sum = 0
    Sum += Y[0]
    Sum += Y[-1]
    
    for i in range(len(Y)-2):
        Sum += 2*Y[i+1]
        
    return h/2*Sum
#%%
#The result of Tapezoidal intergration
h = 6 
X = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84]
Y = [124, 134, 148, 156, 147, 133, 121, 109, 99, 85, 78, 89, 104, 116, 123]
print('In feet:', Ta(Y, 6))
print('In mile:', Ta(Y,6)**0.000189394)
    
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

print('In feet:', Sim(Y, 6))
print('In mile:', Sim(Y, 6)**0.000189394)


# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 09:55:04 2023

@author: joonc
"""

#Numerical Integration

import pandas as pd
import matplotlib.pyplot as plt

#%%
# Given data x = time (s) y = speed (feet/second)
x = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84]
y = [124, 134, 148, 156, 147, 133, 121, 109, 99, 85, 78, 89, 104, 116, 123]

data = {
        'time' : x,
        'speed' : y 
        }

df = pd.DataFrame(data)
#%%

#At first, to get the idea approximate for the length of the track in mile.
#length of the track (feet) = speed (feet/second) * time_interval (second)

Sf = 0
for i in range(len(df)):
    a = 6
    b = df.iloc[i]['speed']
    c = a*b
    Sf += c

print('nearly:',Sf,'feet')

#change the units feet* 0.000189394 (miles/feet)
# 1 feet = 0.000189394 miles
Sm = Sf*0.000189394
print('nearly:',Sm,'miles')
#%%
#Use the given data point
#Choose the numerical integration method

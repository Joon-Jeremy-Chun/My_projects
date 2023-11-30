# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 15:11:10 2023

@author: joonc
"""

#%%
#Use the given data points
#Choose the numerical integration method - Midpoint Rule
#Consider odd data points as midpoints
#then, h will be twice. h = 2*(b-a)/n

def Mid(Y, h):
    Sum = 0
    
    #Midpoint Rule need odd terms
    if len(Y)%2 == 0:
        print("cannot apply the Simpon's Rule; even # terms")
        return None
    
    else:    
        for i in range(len(Y)):
            if i%2 != 0: #odd terms
                Sum += 1*Y[i]
            else: #even terms
                None
                        
        return 2*h*Sum
    
#%%
#Calculate the Simpson's Rule

X = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84]
Y = [124, 134, 148, 156, 147, 133, 121, 109, 99, 85, 78, 89, 104, 116, 123]

print("-Simpson's Rule-")
print('In feets:', Mid(Y, 6))
print('In miles:', Mid(Y, 6)*0.000189394 if isinstance(Mid(Y, 6), (int, float)) else "None")
#%%


# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 15:11:10 2023

@author: joonc
"""
#Use the given data points
#Choose the numerical integration method - Midpoint Rule
#%%
#Consider all odd data points as midpoints
#Then, h will be twice. h = 2*(b-a)/n

def Mid(Y, h):
    Sum = 0
    
    #Midpoint Rule need odd terms
    if len(Y)%2 == 0:
        print("cannot apply the Midpoint Rule; even # terms")
        return None
    
    else:    
        for i in range(len(Y)):
            if i%2 != 0: #odd terms
                Sum += 1*Y[i]
            else: #even terms
                None
                        
        return 2*h*Sum # need *2
    
#%%
#Calculate the Simpson's Rule
#length of the track (feet) = speed (feet/second) * time_interval (second)

X = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84]
Y = [124, 134, 148, 156, 147, 133, 121, 109, 99, 85, 78, 89, 104, 116, 123]

print("-Midpoint Rule-")
print('In feets:', Mid(Y, 6))
print('In miles:', Mid(Y, 6)*0.000189394 if isinstance(Mid(Y, 6), (int, float)) else "None")
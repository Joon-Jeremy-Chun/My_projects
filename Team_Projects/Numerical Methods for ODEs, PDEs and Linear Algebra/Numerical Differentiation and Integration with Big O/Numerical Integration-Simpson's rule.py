# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:11:09 2024

@author: joonc
"""
#Objectives: To Build Simpson's Rule integration
import numpy as np

#%%
#Define the Simpson's Rule integration (for continous functions)
#F : functions, a: starting point, b: Ending point, N: # of segments 
def SimC(F, a, b, N):
    Sum = 0  
    h = (b - a) / N
    
    #Simpson's Rule need odd terms, and a > b
    if a >= b:
        print("cannot apply the inegration; a >= b")
        return None
    
    if N%2 == 0:
        print("cannot apply the Simpon's Rule; even # of terms")
        return None
        
    else:
        for i in range(N):
            if i%2 != 0:
                Sum += 4*F(a + i*h)
            else:
                Sum += 2*F(a + i*h)
        #the First and Last terms correction
        Sum += -F(a)
        Sum += -F(b)
        
        return h/3*Sum
        
#%%
#Define the text function
def T_Sin(x):
        return np.sin(x)
#%%
#Define the ranges of x and N
Y = SimC(T_Sin, 0, np.pi, 101)
print(Y)


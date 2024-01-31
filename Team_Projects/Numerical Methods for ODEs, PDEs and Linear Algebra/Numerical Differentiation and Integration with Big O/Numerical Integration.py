# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:11:09 2024

@author: joonc
"""
import numpy as np

#Simpson's Rule
#%%
#Define the Simpson's Rule integration (for data points)
def Sim(Y, h):
    Sum = 0
    
    #Simpson's Rule need odd terms
    if len(Y)%2 == 0:
        print("cannot apply the Simpon's Rule; even # terms")
        return None
    
    else:    
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
#Define the Simpson's Rule integration (for continous functions)
#F : functions, a: starting point, b: Ending point, h: length of each segments 
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
        Sum += -F[i]
        Sum += -F[i]
        
        return h/3*Sum
        
#%%
def T_Sin(x):
    
    return np.sin(x)
#%%
SimC(T_Sin, 0, np.pi, 1)

#%%
T_Sin(3.14)
SimC(np.sin(), 1, np.pi, 1)

 #%%
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 17:49:30 2024

@author: joonc
"""

#%%
#F : functions, a: starting point, b: Ending point, N: # of Subinvervals
def SimC(F, a, b, N):
    Sum = 0
    h = (b - a) / N
    
    #Simpson's Rule needs odd terms, and a > b
    if a >= b:
        print("cannot apply the integration; a >= b")
        return None
    
    if N%2 != 0:
        print("cannot apply the Simpon's Rule; even # of terms")
        return None
        
    else:
        for i in range(1,N): #from 1 up to (not including N)
            if i%2 != 0:
                Sum += 4*F(a + i*h)
            else:
                Sum += 2*F(a + i*h)             
        Sum += F(a) + F(b) #the First and Last terms correction

        return h/3*Sum
#%%

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:09:45 2024

@author: joonc
"""
import numpy as np
#%%
def FiveP(F, x_0, h):
    Sum = 0  
        
    #It needs h >= 0
    if h <= 0:
        print("cannot apply; h <= 0")
        return None
           
    else:
        Sum += +F(x_0 - 2*h)
        Sum += -8*F(x_0 - h)
        Sum += +8*F(x_0 + h)
        Sum += -F(x_0 + 2*h)
        
        return 1 / (12*h) * Sum
    
#%%
def T_Sin(x):
    return np.sin(x)
#%%
result = FiveP(T_Sin, 1, 0.001)
print(result)
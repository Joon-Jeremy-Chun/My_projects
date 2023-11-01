# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 20:45:54 2023

@author: joonc
"""

import numpy as np
import pandas as pd

def secant(f, x_0, x_1, diff_a_b, max_iter):
    if f(x_0) - f(x_1) == 0:
        print("secant method may not converge because f(x_n) - f(x_n-1) = 0")
        return None
    
    data_list = []
    iteration = 0
    
    x_2 = 0
    a = x_0
    b = x_1
    c = x_2
    
    data_x_iter = {
        "iteration": 0, "x_" : 0, "x_value" : a
    }
    data_list.append(data_x_iter)
    data_x_iter = {
        "iteration": 0, "x_" : 1, "x_value" : b
    }
    data_list.append(data_x_iter)
    
    while abs(b - a) > diff_a_b and iteration < max_iter and f(b)-f(a) != 0:
        
        iteration += 1
        print("~~iteration :", iteration)

        if f(c) == 0:
            print("root :", c)
            return c

        else:
            c = b - (f(b)*(b - a))/(f(b) - f(a))
            print("x:", c)  
            
            a = b
            b = c
            
            data_x_iter = {
                "iteration": iteration, "x_" : iteration + 2, "x_value" : c
            }
            data_list.append(data_x_iter)
        
        print("\n")
    return pd.DataFrame(data_list)

# Define the function that want to find the root of
def f(x):
   return x**2-3


# Initial x_0 and x_1
x_0 = -1
x_1 = 2
diff_a_b = 1e-20
#max_iterations = 10

result = secant(f, x_0, x_1, diff_a_b, 100)
print(result)

y = np.sqrt(3) 
print(y)
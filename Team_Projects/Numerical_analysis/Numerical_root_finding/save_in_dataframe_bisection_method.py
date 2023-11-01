# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 17:52:02 2023

@author: joonc
"""

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def bisection(f, a, b, diff_a_b, max_iter):
    if f(a) * f(b) >= 0:
        print("Bisection method may not converge because f(a) * f(b) >= 0")
        return None
    
    data_list = []
    iteration = 0
    
    data_a_b_iter = {
        "iterations": 0, "a": a, "b": b
    }
    data_list.append(data_a_b_iter)
    
    
    while (b - a) > diff_a_b and iteration < max_iter:
        midpoint = (a + b) / 2.0

        iteration += 1

        if f(midpoint) == 0:
            print("midpoint",midpoint)
            return midpoint

        if f(a) * f(midpoint) < 0:
            b = midpoint
            print("a:", a, "b:", midpoint)
            data_a_b_iter = {
                "iterations": iteration, "a": a, "b": midpoint
            }
            data_list.append(data_a_b_iter)
            
        else:
            a = midpoint
            print("a:", midpoint, "b:", b) 
            data_a_b_iter = {
                "iterations": iteration, "a": midpoint, "b": b
            }
            data_list.append(data_a_b_iter)
        
        print("\n")
#    return midpoint, iteration
#    return (a + b) / 2.0, iteration
    return pd.DataFrame(data_list)


# Define the function that want to find the root of
def f(x):
#   return x**2-3
   return np.sin(x)


# Initial interval [a, b] and diff_a_b(tolerance)
a = -1
b = 2
diff_a_b = 1e-6
#max_iterations = 10

result = bisection(f, a, b, diff_a_b, 100)

print(result)
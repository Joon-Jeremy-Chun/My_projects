# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 17:52:02 2023

@author: joonc
"""

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

#Define the method as a function
#Outcome: the dataframe; a and b values, its iteration
#f: function, a: left end point, b: right end point, diff_a_b: difference a and b, max_iter: maximum # of iteration
#To prevent infinite loops, limited by 1.difference between a and b, and 2.the number of iterations.
#also difference between a and b can be considered as MAX error in that iteration. ( == accuray)

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
# def f(x):
# #   return x**2-3
#    return np.sin(x)

def f(x):
   return 4*np.pi*(x+0.25)+(-2000/(x**2))+(-2*0.25*1000)/(np.pi*x**3)

# Initial interval [a, b] and diff_a_b(tolerance)
a = 5
b = 6
diff_a_b = 1e-4
#max_iterations = 10

result = bisection(f, a, b, diff_a_b, 100)

print(result)
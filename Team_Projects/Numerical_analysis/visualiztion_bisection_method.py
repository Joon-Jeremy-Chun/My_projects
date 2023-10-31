# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 17:42:37 2023

@author: joonc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def bisection(f, a, b, decimal, max_iter):
    if f(a) * f(b) >= 0:
        print("Bisection method may not converge because f(a) * f(b) >= 0")
        return None
    
    data_list = []
    iteration = 0
    
    data_a_b_iter = {
        "iterations": 0, "a": a, "b": b
    }
    data_list.append(data_a_b_iter)
    
    
    while (b - a) / 2.0 > decimal and iteration < max_iter:
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


# Define the function you want to find the root of
def f(x):
   return x**2-3
#   return np.sin(x)


# Initial interval [a, b] and decimal(tolerance)
a = -1
b = 2
decimal = 1e-6
#max_iterations = 10

result = bisection(f, a, b, decimal, 100)


#print(result)

df = result

#root by calculating
#later

# Create a line plot for X- 'a' and 'b' values and Y- 'iterations'
plt.figure(figsize=(10, 5))
plt.plot(df['iterations'], df['a'], label='a', marker='o', markersize=2)
plt.plot(df['iterations'], df['b'], label='b', marker='s', markersize=2)
plt.xlabel('Iterations')
plt.ylabel('Value')
plt.legend()
plt.title('Plot of a and b values over iterations')
plt.grid(True)

plt.axhline(y=0.5, color='red', linestyle='--', label='Horizontal Line at the root')


plt.show()

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
#   return x**2-3
   return np.sin(x)


# Initial interval [a, b] and decimal(tolerance)
a = -1
b = 2
decimal = 1e-6
#max_iterations = 10

result = bisection(f, a, b, decimal, 100)


#print(result)

# Sample DataFrame
# data = {
#     'iterations': list(range(22)),
#     'a': [-1.0, -1.0, -0.25, -0.25, -0.0625, -0.0625, -0.015625, -0.015625, -0.00390625, -0.00390625,
#           -0.0009765625, -0.0009765625, -0.000244140625, -0.000244140625, -6.103515625e-05, -6.103515625e-05,
#           -1.52587890625e-05, -1.52587890625e-05, -3.814697265625e-06, -3.814697265625e-06, -9.5367431640625e-07, -9.5367431640625e-07],
#     'b': [2.0, 0.5, 0.5, 0.125, 0.125, 0.03125, 0.03125, 0.0078125, 0.0078125, 0.001953125,
#           0.001953125, 0.00048828125, 0.00048828125, 0.0001220703125, 0.0001220703125, 0.000030517578125,
#           0.000030517578125, 0.00000762939453125, 0.00000762939453125, 0.0000019073486328125, 0.0000019073486328125, 0.000000476837158203125]
# }

df = result

# Create a line plot for 'a' and 'b' values against 'iterations'
plt.figure(figsize=(10, 5))
plt.plot(df['iterations'], df['a'], label='a', marker='o', markersize=2)
plt.plot(df['iterations'], df['b'], label='b', marker='s', markersize=2)
plt.xlabel('Iterations')
plt.ylabel('Value')
plt.legend()
plt.title('Plot of a and b values over iterations')
plt.grid(True)

plt.show()

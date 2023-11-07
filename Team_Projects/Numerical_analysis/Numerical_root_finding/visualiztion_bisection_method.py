# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 17:42:37 2023

@author: joonc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Define the method as a function
#Outcome: the dataframe; #iterations, a, and b values
#f: function, a: left end point, b: right end point, diff_a_b: difference a and b, max_iter: maximum # of iteration
#To prevent infinite loops, limited by 1.difference between a and b, and 2.the number of iterations.
#also difference between a and b can be considered as MAX error in that iteration. ( == accuray)
def bisection(f, a, b, diff_a_b, max_iter):
    if f(a) * f(b) >= 0:
        print("Bisection method may not converge because f(a) * f(b) >= 0")
        return None
    
    data_list = []
    iteration = 0
    #record given a,b value as a initial value
    data_a_b_iter = {
        "iterations": 0, "a": a, "b": b
    }
    data_list.append(data_a_b_iter)
    
    
    while abs(b - a) > diff_a_b and iteration < max_iter:
        #midoint is the tentative point can be a or b in next cycle
        midpoint = (a + b) / 2.0

        iteration += 1

        if f(midpoint) == 0:
            print("midpoint",midpoint)
            return midpoint

        if f(a) * f(midpoint) < 0:
            b = midpoint
#            print("a:", a, "b:", midpoint)
            data_a_b_iter = {
                "iterations": iteration, "a": a, "b": midpoint
            }
            data_list.append(data_a_b_iter)
            
        else:
            a = midpoint
#            print("a:", midpoint, "b:", b) 
            data_a_b_iter = {
                "iterations": iteration, "a": midpoint, "b": b
            }
            data_list.append(data_a_b_iter)
        
#        print("\n")
    return pd.DataFrame(data_list)

# Define the function that want to find the root of
def f(x):
   return 4*np.pi*(x+0.25)+(-2000/(x**2))+(-2*0.25*1000)/(np.pi*x**3)

# Initial interval [a, b] and diff_a_b(tolerance)(== accuray)
a = 1
b = 100
diff_a_b = 1e-6
#max_iterations = 10

result = bisection(f, a, b, diff_a_b, 10000)

# drowing plots in order to check convergence and speed of convergence

df = result

# Create a line plot for Y- 'a' and 'b' values and X- # iteration

plt.plot(df['iterations'], df['a'], label='a:left point', marker='o', markersize=2)
plt.plot(df['iterations'], df['b'], label='b:right point', marker='s', markersize=2)
plt.xlabel('Iterations')
plt.ylabel('Value')
plt.legend()
plt.title('Bisection method : Plot of a and b values over iterations')
plt.grid(True)

#y-value temporally solve by hand
#plt.axhline(y=np.sqrt(3), color='red', linestyle='--', label='Horizontal Line at the root')


#plt.show()
plt.savefig('Bisection_method.png')
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 21:18:53 2023

@author: joonc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# define the method as a function
# output of the function is a dataframe
# f: function, x_0: first given point, x_1: second given point, 
# diff_a_b: difference between x_n and x_n_1 or left and right points, max_iter: limited by # of maximum iteration for loof (prevent infinit roof)
def secant(f, x_0, x_1, diff_a_b, max_iter):
    if f(x_0) - f(x_1) == 0:
        print("secant method may not converge because f(x_n) - f(x_n-1) = 0")
        return None
    
    data_list = []
    iteration = 0
    
    x_2 = 0.123 # define any sudo x_2 value at first that avoid for f(x_2) not to undefined
    a = x_0 # let a is the first in a cycle of the roof
    b = x_1 # let b is the second in a cycle of the roof
    c = x_2 # let c is the third in a cycle of the roof
    
    # since x_0 and x_1 are given. record first
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
#        print("~~iteration :", iteration)
        
        #If we find the tageted root then must stop the roof and gives of output.
        if f(c) == 0:
            print("root :", c)
            return c
        
        # c is x_n, we need to calculate, b is one before the value, c is two before the value
        else:
            c = b - (f(b)*(b - a))/(f(b) - f(a))
#           print("x:", c)  
            
            #to prepare next cycle let b become a and c become b
            a = b
            b = c
            
            #record the value x_n
            data_x_iter = {
                "iteration": iteration, "x_" : iteration + 2, "x_value" : c
            }
            data_list.append(data_x_iter)
        
#        print("\n")
    return pd.DataFrame(data_list)

#define the f(x) we want to find the root.
def f(x):
   return 4*np.pi*(x+0.25)+(-2000/(x**2))+(-2*0.25*1000)/(np.pi*x**3)

# Initial point; x_0 and x_1
x_0 = 5
x_1 = 6
# The maximum difference between x_n and x_n_1
diff_a_b = 1e-4
#max_iterations = 10

#Output save in the 'result' variable
result = secant(f, x_0, x_1, diff_a_b, 100)


# drowing plots in order to check convergence and speed of convergence

df = result

# Create a line plot for X- # iteration and and Y- 'x_n' valuse (converge to the root)

plt.figure(figsize=(10, 5))
plt.plot(df['iteration'], df['x_value'], label='x_n', marker='o', markersize=2, color = 'g')
plt.xlabel('x_')
plt.ylabel('Value')
plt.legend()
plt.title('Secant method : Plot of x_n values over iterations')
plt.grid(True)

plt.show()
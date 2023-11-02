# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 17:47:46 2023

@author: joonc
"""

import numpy as np

def secant(f, x_0, x_1, diff_a_b, max_iter):
    if f(x_0) - f(x_1) == 0:
        print("secant method may not converge because f(x_n) - f(x_n-1) = 0")
        return None

    iteration = 0
    x_2 = 0.123 # define any x_2 value at first that avoid for(x_2) to undefined
    a = x_0
    b = x_1
    c = x_2
    
    while abs(b - a) > diff_a_b and iteration < max_iter and f(b)-f(a) != 0 and c !=0:
        
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
            
        
        print("\n")
    return c, iteration


# Define the function that want to find the root of
# def f(x):
#     return x**2-3

def f(x):
   return 4*np.pi*(x+0.25)+(-2000/(x**2))+(-2*0.25*1000)/(np.pi*x**3)

# Initial x_0 and x_1
x_0 = 5
x_1 = 6
diff_a_b = 1e-4
#max_iterations = 10

result = secant(f, x_0, x_1, diff_a_b, 100)
print(result)

# y = np.sqrt(3) 
# print(y)

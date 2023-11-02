# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 17:00:02 2023

@author: joonc
"""
#pip install matplotlib

import numpy as np

#To prevent infinite loops, limit by 1.difference between a and b, and 2.the number of iterations.
 
def bisection(f, a, b, diff_a_b, max_iter):
    if f(a) * f(b) >= 0:
        print("Bisection method may not converge because f(a) * f(b) >= 0")
        return None

    iteration = 0
    while (b - a) > diff_a_b and iteration < max_iter:
        midpoint = (a + b) / 2.0

        iteration += 1
        print("~~iteration :", iteration)

        if f(midpoint) == 0:
            print("midpoint",midpoint)
            return midpoint

        if f(a) * f(midpoint) < 0:
            b = midpoint
            print("a:", a, "b:", midpoint)
        else:
            a = midpoint
            print("a:", midpoint, "b:", b)       
        
        print("\n")
    return midpoint, iteration
#    return (a + b) / 2.0, iteration


# Define the function that want to find the root of
# def f(x):
#    return x**2-3
#   return np.sin(x)

def f(x):
    return 4*np.pi*(x+0.25)+(-2000/(x**2))+(-2*0.25*1000)/(np.pi*x**3)

# Initial interval [a, b] and diff_a_b(tolerance)
a = 5
b = 6
diff_a_b = 1e-4
#max_iterations = 10

result = bisection(f, a, b, diff_a_b, 100)
print(result)

#real root 
# y=np.sqrt(3)
# print(y)
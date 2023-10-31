# -*- coding: utf-8 -*-

#pip install matplotlib

import numpy as np
import matplotlib.pyplot as plt

def bisection(f, a, b, accuracy, max_iter):
    if f(a) * f(b) >= 0:
        print("Bisection method may not converge because f(a) * f(b) >= 0")
        return None

    iteration = 0
    while (b - a) / 2.0 > accuracy and iteration < max_iter:
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


# Define the function you want to find the root of
def f(x):
#   return x**2-3
   return np.sin(x)


# Initial interval [a, b] and tolerance
a = -1
b = 2
accuracy = 1e-6
#max_iterations = 10

result = bisection(f, a, b, accuracy, 100)

#if result is not None:
#    print(f"Approximate root: {result:.5f}")
#else:
#    print("Bisection method did not converge within the specified iterations.")
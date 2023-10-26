# -*- coding: utf-8 -*-

#pip install matplotlib

import numpy as np
import matplotlib.pyplot as plt

def bisection(f, a, b, tol, max_iter):
    if f(a) * f(b) >= 0:
        print("Bisection method may not converge because f(a) * f(b) >= 0")
        return None

    iteration = 0
    while (b - a) / 2.0 > tol and iteration < max_iter:
        midpoint = (a + b) / 2.0

        if f(midpoint) == 0:
            return midpoint

        if f(a) * f(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint

        iteration += 1

    return (a + b) / 2.0

# Define the function you want to find the root of
def f(x):
    return np.sin(x)

# Initial interval [a, b] and tolerance
a = 0
b = 4
tolerance = 1e-5
max_iterations = 100

result = bisection(f, a, b, tolerance, max_iterations)

if result is not None:
    print(f"Approximate root: {result:.5f}")
else:
    print("Bisection method did not converge within the specified iterations.")
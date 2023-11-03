# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 18:30:00 2023

@author: joonc
"""

import numpy as np

# Hybrid method - bisection + secant

def hybrid_bi_se(f, a, b, diff_a_b, max_iter):
    if f(b) - f(a) == 0:
        print("Hybrid method may not converge because f(b) - f(a) = 0")
        return None

    if f(a) * f(b) > 0:
        print("Warning - Hybrid method may not guarantee finding the root because f(a) * f(b) > 0")
        return None

    iteration = 0
    x_n_1 = (a + b) / 2 
    ax = a
    bx = b

    while abs(b - a) > diff_a_b and iteration < max_iter and f(b) - f(a) != 0:

        iteration += 1

        # x_n is the x_n which is calculated every cycle
        print("~~Iteration:", iteration)
        a = ax
        b = bx
        x_n = x_n_1



        if f(x_n) == 0:
            print("Root:", x_n)
            return x_n

        if a <= x_n <= b and f(a) * f(x_n) < 0:
            x_n_1 = x_n - (f(x_n) * (x_n - a)) / (f(x_n) - f(a))
            print("a:", a, "x:", x_n, "b:", b, "1")
            ax = a
            bx = x_n

        if a <= x_n <= b and f(x_n) * f(b) < 0:
            x_n_1 = x_n - (f(x_n) * (x_n - b)) / (f(x_n) - f(b))
            print("x:", x_n, "2")
            ax = x_n
            bx = b

        if x_n_1 <= a or b <= x_n:
            x_n = (a + b) / 2
            print("x:", x_n, "3")
            ax = a
            bx = b
            
        else:
            print("Error")
            print("a:", a, "x_n:", x_n, "b:", b)
            print("f(a):", f(a), "f(x_n):", f(x_n), "f(b):", f(b))
        
        print("\n")

    return x_n, iteration

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
result = hybrid_bi_se(f, x_0, x_1, diff_a_b, 20)
print(result)
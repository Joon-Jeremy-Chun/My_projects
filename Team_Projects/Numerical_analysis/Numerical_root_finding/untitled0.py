# -*- coding: utf-8 -*-

import numpy as np
from sympy import *
import math
from scipy.optimize import minimize

# To prevent infinite loops, limit by 1.difference between a and b, and 2.the number of iterations.

def hybrid(f, a, b, diff_a_b, max_iter):
# Condition to proceed with iterations
    if f(a) * f(b) >= 0:
        print("hybrid method may not converge because f(a) * f(b) >= 0")
        return None

    iteration = 0
    while (b - a) > diff_a_b and iteration < max_iter:
        # Perform secant method first
        x_prime = b - ((f(b)*(b-a))/(f(b)-f(a)))

        #Denote iteration
        iteration += 1
        print("~~iteration :", iteration)

        #Check for location of new x value
        #First condition checks if new value lies outside of the interval
        if x_prime < a or x_prime>b:
            #Performs bisection to obtain new x value
            x_prime = (b-a)/2.0

            #Checks image of new x value
            if f(a) * f(x_prime) < 0:
                b = x_prime
                print("a:", a, "b:", x_prime)
            else:
                a = x_prime
                print("a:", x_prime, "b:", b)

        else:
            #Checks image of new x value if secant lies within the interval
            if f(a)*f(x_prime)<0:
                b = x_prime
                print("a:", a, "b:", x_prime)
            else:
                a=x_prime
                print("a:", x_prime, "b:", b)

        print("\n")
    height = h(x_prime)
    return x_prime, iteration,height

# Define the function that want to find the root of
def f(x):
    return (-2000/(x**2))-(500/((math.pi)*(x**3)))+(4*math.pi*x)+math.pi
def h(c):
    return 1000/(math.pi*c**2)


# Initial interval [a, b] and diff_a_b(tolerance)
c = input("Please input an endpoint for your interval: ", 5)
d = input("Please input another endpoint for your interval: ", 6)
c=float(c)
d=float(d)

a = min([c,d])
b = max([c,d])

a = 5 
b = 6
diff_a_b = 1e-4
# max_iterations = 100
#Print final number of iterations required to land within the error
result = hybrid(f, a, b, diff_a_b, 100)
print(result)
# if result[1]<100:
#     print(f"We require {result[1]} iterations for the error to lie within 10e-4.")
# else:
#     print(f"We require over 100 iterations for the error to lie within 10e-4.")
# print(f"The most optimal radius of the can will be {round(result[0],2)} cm and the most optimal height will be  {round(result[2],2)} cm.")
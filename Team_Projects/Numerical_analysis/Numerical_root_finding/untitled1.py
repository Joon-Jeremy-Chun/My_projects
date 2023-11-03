# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 09:45:28 2023

@author: joonc
"""
import math
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
 #   height = h(x_prime)
    return x_prime, iteration #height

def f(x):
    return (-2000/(x**2))-(500/((math.pi)*(x**3)))+(4*math.pi*x)+math.pi

a = 5 
b = 6
diff_a_b = 1e-4
# max_iterations = 100
#Print final number of iterations required to land within the error
result = hybrid(f, a, b, diff_a_b, 100)
print(result)
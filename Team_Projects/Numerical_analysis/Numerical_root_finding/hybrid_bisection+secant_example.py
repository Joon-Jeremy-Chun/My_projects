# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 08:51:17 2023

@author: joonc
"""
#hybrid method practice with the textbook example

#hybrid method
#bisection + secant

import numpy as np

#hybrid method = bisection + secant
#hybrid function input define
# 1.f; function, for radious 2.g; function, for hight 3. a; left point 4. b; right point
# 5. diff_a_b; in this cods difference between x_n_+1 and x_n 6. max_iter; bounced max iteration
# note - we are define the error (accuracy) bounced by diff_a_b. because it is cauchy sequence.

#hybrid function output define 
# 1. r; radious 2. iteration 3. h; hight

def hybrid(f, g, a, b, diff_a_b, max_iter):
# Condition to proceed with iterations
    if f(a) * f(b) >= 0:
        print("hybrid method may not converge because f(a) * f(b) >= 0")
        return None
    
    if f(b) - f(a) == 0 :
        print("hybrid method may not converge becasue f(b) - f(a) = 0")
        return None

    iteration = 0
    x_prime = (a+b)/2 #initiated the first sudo x_prime (not record, not effect) for geting start the first roof
    
    while abs(x_prime - (b - ((f(b)*(b-a))/(f(b)-f(a)))) ) > diff_a_b and iteration < max_iter and f(b) - f(a) != 0:
        # Perform secant method first
        x_prime = b - ((f(b)*(b-a))/(f(b)-f(a)))

        #Denote iteration
        iteration += 1
        print("~~iteration :", iteration)

        #Check for location of new x value
        #First condition checks if new value lies outside of the interval
        if x_prime < a or x_prime > b:
            #Performs bisection to obtain new x value
            print("bicetion mothod")
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
            print("secant mothod")
            if f(a)*f(x_prime)<0:
                a = a
                b = x_prime
                
                print("a:", a, "b:", x_prime)
            else:
                a = x_prime
                b = b
                print("a:", x_prime, "b:", b)

        print("\n")
    height = g(x_prime)
    return x_prime, iteration, height

#define the function we want to know
# def f(x):
#     return 4*np.pi*(x+0.25)+(-2000/(x**2))+(-2*0.25*1000)/(np.pi*x**3)
# def g(c):
#     return 1000/(np.pi*c**2)

# def f(x):
#    return (20*x -1) / (19*x)

def f(x):
    return 2*x*np.exp(-15) -2*np.exp(-15*x) +1


def g(x):
    return x
a = 0
b = 1
diff_a_b = 1e-20
# max_iterations = 100
#Print final number of iterations required to land within the error
result = hybrid(f, g, a, b, diff_a_b, 500)
print(result)

a = 0.5
b = 1

x_1 = b - ((f(b)*(b-a))/(f(b)-f(a)))
x_1
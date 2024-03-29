# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 17:00:02 2023

@author: joonc
"""
#pip install matplotlib

import numpy as np

#Define the method as a function
#Outcome: midpoint (root) value, its iteration
#f: function, a: left end point, b: right end point, diff_a_b: difference a and b, max_iter: maximum # of iteration
#To prevent infinite loops, limited by 1.difference between a and b, and 2.the number of iterations.
#also difference between a and b can be considered as MAX error in that iteration. ( == accuray)
def bisection(f, a, b, diff_a_b, max_iter):
    if f(a) * f(b) >= 0:
        print("Bisection method may not converge because f(a) * f(b) >= 0")
        return None

    iteration = 0
    while abs(b - a) > diff_a_b and iteration < max_iter:
        #midoint is the tentative point can be a or b in next cycle
        midpoint = (a + b) / 2.0

        iteration += 1
#        print("~~iteration :", iteration)

        if f(midpoint) == 0:
#            print("midpoint",midpoint)
            return midpoint

        if f(a) * f(midpoint) < 0:
            b = midpoint
#            print("a:", a, "b:", midpoint)
        else:
            a = midpoint
#            print("a:", midpoint, "b:", b)       
        
 #       print("\n")
    return midpoint, iteration


#define the function we want to know
def f(x):
    return 4*np.pi*(x+0.25)+(-2000/(x**2))+(-2*0.25*1000)/(np.pi*x**3)

# Initial interval [a, b] and diff_a_b(tolerance)(== accuray)
a = 1
b = 100
diff_a_b = 1e-8
#max_iterations = 10

result = bisection(f, a, b, diff_a_b, 1000)
print(result)


r = result[0]
h = 1000/(np.pi*(r**2))
SA = 2*np.pi*(r + .25)**2 + (2*np.pi*r + .25)*h
Cost = .5*SA

r = round(r,2)
h = round(h,2)
SA = round(SA, 2)
Cost = round(Cost,2)


print("You optimal dimensions for this can is listed below:")
print(f"Radius = {r}, Height = {h}")
print(f"The total cost to produce one can would be ${Cost}.")
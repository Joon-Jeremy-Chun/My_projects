# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 17:47:46 2023

@author: joonc
"""

import numpy as np

# define the method as a function
# output of the function is a root (the last x_n value of the roof) and # of iteration
# f: function, x_0: first given point, x_1: second given point, 
# diff_a_b: difference between x_n and x_n_1 or left and right points, max_iter: limited by # of maximum iteration for loof (prevent infinit roof)

def secant(f, x_0, x_1, diff_a_b, max_iter):
    if f(x_0) - f(x_1) == 0:
        print("secant method may not converge because f(x_n) - f(x_n-1) = 0")
        return None

    iteration = 0
    x_2 = (x_0 + x_1) / 2 # define any sudo x_2 value at first that avoid for f(x_2) not to undefined
    a = x_0 # let a is the first in a cycle of the roof
    b = x_1 # let b is the second in a cycle of the roof
    c = x_2 # let c is the third in a cycle of the roof
    
    while abs(b - a) > diff_a_b and iteration < max_iter and f(b)-f(a) != 0 and c !=0:
        
        iteration += 1
        print("~~iteration :", iteration)


        #If we find the tageted root then must stop the roof and gives of output.
        if f(c) == 0:
            print("root :", c)
            return c

        else:
            c = b - (f(b)*(b - a))/(f(b) - f(a))
            print("x:", c)  
            
            #to prepare next cycle let b become a and c become b
            a = b
            b = c
            
        
        print("\n")
    return c, iteration


#define the f(x) we want to find the root.
def f(x):
    return 4*np.pi*(x+0.25)+(-2000/(x**2))+(-2*0.25*1000)/(np.pi*x**3)


# Initial x_0 and x_1
x_0 = 5
x_1 = 6
diff_a_b = 1e-4
#max_iterations = 10

result = secant(f, x_0, x_1, diff_a_b, 100)
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

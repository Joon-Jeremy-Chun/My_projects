# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 07:00:17 2023

@author: joonc
"""

#hybrid method
#bisection + secant

import numpy as np

#bisection method
 
def bisection(f, a, b, diff_a_b, max_iter):
    if f(a) * f(b) >= 0:
        print("Bisection method may not converge because f(a) * f(b) >= 0")
        return None

    iteration = 0
    while abs(b - a) > diff_a_b and iteration < max_iter:
        midpoint = (a + b) / 2.0

        iteration += 1
#        print("~~iteration :", iteration)

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

#secant method

def secant(f, x_0, x_1, diff_a_b, max_iter):
    if f(x_0) - f(x_1) == 0:
        print("secant method may not converge because f(x_n) - f(x_n-1) = 0")
        return None

    iteration = 0
    x_2 = 0
    a = x_0
    b = x_1
    c = x_2
    
    while abs(b - a) > diff_a_b and iteration < max_iter and f(b)-f(a) != 0:
        
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

#hybrid method - bisection + secant

def hybrid_bi_se(f, a, b, diff_a_b, max_iter):
    if f(b) - f(a) == 0 :
        print("hybrid method may not converge becasue f(b) - f(a) = 0")
        return None
    
    if f(a)*f(b) > 0 :
        print("warning - hybrid method may not guarantee find the root becasue f(a)*f(b)>0")
        return None
    
    
    iteration = 0
    
    while abs(b - a) > diff_a_b and iteration < max_iter and f(b)-f(a) !=0:
        
        iteration += 1
        x_n_1 = a
        print("~~iteration :", iteration)

        if f(x_n) == 0:
            print("root :", x_n)
            return x_n
        
        if a <= x_n <= b and f(a)*f(x_n) < 0:
            
            x_n_1 = x_n - (f(x_n)*(x_n - a)) / (f(x_n) - f(a))
            print("x :" : x_n_1)
            
        if a <= x_n <= b and f(x_n)*f(b) <0:
            
            x_n_1 = x_n - (f(x_n)*(x_n - b) / (f(x_n) - f(b))
                           
        if x_n <= a or b <= x_n and f(a)*f(b) <0:
            
            x_n_1 = (a + b) / 2

        if f(a) * f(midpoint) < 0:
            b = midpoint
            print("a:", a, "b:", midpoint)
            
        else:
            print("error")
            print("a:", a, "x_n_1:" x_n_1, "b:", b)
            print("f(a):", f(a), "f(x_n_1):", f(x_n_1), "f(b):", f(b))
        
        print("\n")
    return x_n_1, iteration

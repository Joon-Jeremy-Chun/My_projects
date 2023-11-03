# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 07:00:17 2023

@author: joonc
"""

#hybrid method
#bisection + secant

import numpy as np

#hybrid method - bisection + secant

def hybrid_bi_se(f, a, b, diff_a_b, max_iter):
    if f(b) - f(a) == 0 :
        print("hybrid method may not converge becasue f(b) - f(a) = 0")
        return None
    
    if f(a)*f(b) > 0 :
        print("warning - hybrid method may not guarantee to get the root becasue f(a)*f(b)>0")
        return None
    
    
    iteration = 0
    a = a
    b = b
    x_n = 0.123 #any value that abovid undefine the function
    
    while abs(b - a) > diff_a_b and iteration < max_iter and f(b)-f(a) !=0:
        
        iteration += 1
        
        
        #x_2 is the x_n which is that calculate every cycle
        
        print("~~iteration :", iteration)

        if f(x_n) == 0:
            print("root :", x_n)
            return x_n
        
        if a <= x_n <= b and f(a)*f(x_n) < 0:
            
            x_n = x_n - (f(x_n)*(x_n - a)) / (f(x_n) - f(a))
            print("x :", x_n)
            
            a = a
            b = x_n
            
        if a <= x_n <= b and f(x_n)*f(b) < 0:
            
            x_n = x_n - (f(x_n)*(x_n - b) / (f(x_n) - f(b))
            print("root :", x_n)
                        
            a = x_n
            b = b
                           
        if x_n <= a or b <= x_n:
            
            x_n = (a + b) / 2
            print("x :", x_n)
            a = a
            b = b
                        
        else:
            print("error")
            print("a:", a, "x_n_1:" x_n_1, "b:", b)
            print("f(a):", f(a), "f(x_n_1):", f(x_n_1), "f(b):", f(b))
        
        print("\n")
    return x_2, iteration

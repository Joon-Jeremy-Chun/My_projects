# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 07:00:17 2023

@author: joonc
"""

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
    """Safeguarded secant/bisection on a sign-changing finite interval."""
    import math
    if not (math.isfinite(a) and math.isfinite(b) and a < b):
        raise ValueError("Expected finite endpoints a < b.")
    if not math.isfinite(diff_a_b) or diff_a_b <= 0 or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("Positive tolerance and iteration limit are required.")
    fa, fb = f(a), f(b)
    if not (math.isfinite(fa) and math.isfinite(fb)):
        raise ValueError("Endpoint function values must be finite.")
    if fa == 0:
        return a, 0, g(a)
    if fb == 0:
        return b, 0, g(b)
    if (fa > 0) == (fb > 0):
        raise ValueError("The endpoints must bracket a sign change.")
    for iteration in range(1, max_iter + 1):
        width = b-a
        if width <= 2*diff_a_b:
            midpoint = (a+b)/2
            return midpoint, iteration, g(midpoint)
        x = b-fb*width/(fb-fa) if fb != fa else (a+b)/2
        # A central bracket step prevents secant stagnation on steep functions.
        if not math.isfinite(x) or not a+0.1*width <= x <= b-0.1*width:
            x = (a+b)/2
        fx = f(x)
        if not math.isfinite(fx):
            raise ValueError("Nonfinite function value inside the bracket.")
        if fx == 0:
            return x, iteration, g(x)
        if (fa > 0) != (fx > 0):
            b, fb = x, fx
        else:
            a, fa = x, fx
    raise RuntimeError("Hybrid method did not reach the bracket tolerance.")

#define the function we want to know
def f(x):
    return 4*np.pi*(x+0.25)+(-2000/(x**2))+(-2*0.25*1000)/(np.pi*x**3)
def g(c):
    return 1000/(np.pi*c**2)

a = 1
b = 100
diff_a_b = 1e-8
# max_iterations = 100
#Print final number of iterations required to land within the error
result = hybrid(f, g, a, b, diff_a_b, 10000)
print(result)


print(f"The most optimal radius of the can will be {round(result[0],2)} cm and the most optimal height will be  {round(result[2],2)} cm.")
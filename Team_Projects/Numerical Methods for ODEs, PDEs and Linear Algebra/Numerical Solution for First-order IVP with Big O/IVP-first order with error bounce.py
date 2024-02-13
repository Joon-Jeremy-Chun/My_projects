# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:20:34 2024

@author: joonc
"""

# Consider the initial value problem:
#  y′ = 2
#  t y +t2et, 1 ≤ t ≤ 2, y(1) = 0
#  with exact solution y(t) = t2(et − e).
#  The Midpoint method is given by
#  yn+1 = yn−1 +2hf(tn,yn).
#  1. By computing its truncation error, determine the order of accuracy of the Midpoint
#  method.
#  2. Implement the Midpoint method with h = 0.01 to approximate the solution to the
#  above IVP and compare it with the exact solution.
#  3. Plot the approximate solution given by the Midpoint method and the exact solution
#  curve on the same window. Be sure to label your curves.
#  4. Compute the absolute error at each time step. Your output should include the print
#  out of approximate value, exact value, and absolute error at each time step.
#  5. Use your code to verify the order of accuracy of the Midpoint method (see notes on
#  Order of Accuracy on how to do this) and compare it with your theoretical result
#  obtained in #1.
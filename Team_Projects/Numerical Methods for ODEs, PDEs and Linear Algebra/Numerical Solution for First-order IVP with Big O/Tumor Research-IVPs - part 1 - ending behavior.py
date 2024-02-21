# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 22:47:57 2024

@author: Matthew Perez
"""
from matplotlib import pyplot as plt
import math
# math.sqrt(n) gives you the square root of n
import numpy as np

# creating all of our parameter values
l = 1
m = 1
a = .25
S = .8
sigma = .25
h = 1/16

# vector to store approximations, with I.C. R(0) = a
approx = [a]
# vector to store time steps
t_vec = []

# values that will be stored in t_vec
for i in range(0, 251):
    t_vec.append(i)

# creating our Tumor Growth Rate Model as a function
def tumor(r, l, m, S, sigma):
    left = (-1/3)*S*r
    right_top = (2*l*sigma)
    right_bottom = ((m*r)+math.sqrt((m**2)*(r**2)+(4*sigma)))
    value = left + (right_top/right_bottom)
    return value

# RK-4 Method
for j in range(0,len(t_vec)):
    if j == 0:
        print(f"The initial condition for your tumor growth model is R(0) = {approx[0]}.\n")
    else:
        k_1 = h*tumor(approx[j-1], l, m, S, sigma)
        k_2 = h*tumor(approx[j-1] + (k_1/2), l, m, S, sigma)
        k_3 = h*tumor(approx[j-1] + (k_2/2), l, m, S, sigma)
        k_4 = h*tumor(approx[j-1] + k_3, l, m, S, sigma)
        r_approx = approx[j-1] + (1/6)*(k_1 + 2*k_2 + 2*k_3 + k_4)
        approx.append(r_approx)

print(f"Tumor radius R(t) approaches {round(approx[-1],4)} as t -> infinity.")

# Plotting our solution curve R(t)
plt.plot(t_vec, approx, label='R(t)')
plt.xlabel('Time (t)')
plt.ylabel('Radius (r)')
plt.title('Tumor Growth Over Time')
plt.legend()
plt.show()


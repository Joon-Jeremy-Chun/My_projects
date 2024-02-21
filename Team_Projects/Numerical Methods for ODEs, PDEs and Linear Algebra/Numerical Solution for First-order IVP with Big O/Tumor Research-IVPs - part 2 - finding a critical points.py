# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 22:50:41 2024

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
h = 1/16

# vector to store ALL sigma values
sigma_vec = [0]
# vector to store sigma values w/ decreasing tumor size
tumor_decrease = []

# values that will be stored in our sigma_vec
# we know that at sigma = .25, we reach a leveling off point of .8598
# Thus, we can divided .25/.001 to acquire the maximum number of sigmas we need to test to find
# our sigma_2 value. sigma_1 = 0 since we cannot have a negative nutrient level
for y in range(0, 251):
    if y == 0:
        print("The lowest possible value for sigma we can have is 0.")
    else:
        value = sigma_vec[y - 1] + .001
        value = round(value, 3)
        sigma_vec.append(value)
#%%
# creating our Tumor Growth Rate Model as a function
def tumor(r, l, m, S, sigma):
    left = (-1/3)*S*r
    right_top = (2*l*sigma)
    right_bottom = ((m*r)+math.sqrt((m**2)*(r**2)+(4*sigma)))
    value = left + (right_top/right_bottom)
    return value
#%%
# RK-4 Method w/ sigma for loop attached
# since we want to see an immediate decrease in tumor radius, we just need to look at the first term
# of each corresponding sigma value to gather our interval
for w in sigma_vec:
    sigma = w
    k_1 = h*tumor(a, l, m, S, sigma)
    k_2 = h*tumor(a + (k_1/2), l, m, S, sigma)
    k_3 = h*tumor(a + (k_2/2), l, m, S, sigma)
    k_4 = h*tumor(a + k_3, l, m, S, sigma)
    r_approx = a + (1/6)*(k_1 + 2*k_2 + 2*k_3 + k_4)
    if r_approx < a:
        tumor_decrease.append(sigma)

print(f"\nFrom our computations, we can see that our interval for \nstrictly decreasing tumor radii is"
      f" [{tumor_decrease[0]}, {tumor_decrease[-1]}].\n\n"
      f"Tumor decrease when {tumor_decrease[0]} <= sigma <= {tumor_decrease[-1]}.\n")

#%%
# We use this next part of the script to gather all of our data for our plots
tumor_final_sigma = [tumor_decrease[-1], tumor_decrease[-1] + .001]

# vector to store sigma_2 approximations, with I.C. R(0) = a
sigma_2_approx = [a]
# vector to store sigma_2 + .001 approximation with I.C R(0) = a
sigma_increase = [a]
# vector to store time steps
t_vec = []

# values that will be stored in t_vec
for i in range(0, 251):
    t_vec.append(i)

# RK-4 Method for sigma = .021
for u in range(0, len(t_vec)):
    if u == 0:
        print()
    else:
        k_1 = h*tumor(sigma_2_approx[u-1], l, m, S, tumor_final_sigma[0])
        k_2 = h*tumor(sigma_2_approx[u-1] + (k_1/2), l, m, S, tumor_final_sigma[0])
        k_3 = h*tumor(sigma_2_approx[u-1] + (k_2/2), l, m, S, tumor_final_sigma[0])
        k_4 = h*tumor(sigma_2_approx[u-1] + k_3, l, m, S, tumor_final_sigma[0])
        r_approx = sigma_2_approx[u-1] + (1/6)*(k_1 + 2*k_2 + 2*k_3 + k_4)
        sigma_2_approx.append(r_approx)

# RK-4 Method for sigma = .022
for f in range(0, len(t_vec)):
    if f == 0:
        print()
    else:
        k_1 = h*tumor(sigma_increase[f-1], l, m, S, tumor_final_sigma[-1])
        k_2 = h*tumor(sigma_increase[f-1] + (k_1/2), l, m, S, tumor_final_sigma[-1])
        k_3 = h*tumor(sigma_increase[f-1] + (k_2/2), l, m, S, tumor_final_sigma[-1])
        k_4 = h*tumor(sigma_increase[f-1] + k_3, l, m, S, tumor_final_sigma[-1])
        r_approx = sigma_increase[f-1] + (1/6)*(k_1 + 2*k_2 + 2*k_3 + k_4)
        sigma_increase.append(r_approx)



# Plotting both of the solutions onto one graph
plt.plot(t_vec, sigma_2_approx, label='Sigma = .021')
plt.plot(t_vec, sigma_increase, label='Sigma = .022')
plt.xlabel('Time (t)')
plt.ylabel('Radius (r)')
plt.title('Tumor Growth Comparison')
plt.legend()
plt.show()

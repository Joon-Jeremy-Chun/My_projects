# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 16:26:45 2024

@author: Matthew Perez
"""

# import numpy as np
# import matplotlib.pyplot as plt

# # Creates a function that computes Five-Point Formula (1)
# def five_step(f, x_0, h):
#     numerator = f(x_0 - 2*h) - 8*f(x_0 - h) + 8*f(x_0 + h) - f(x_0 + 2*h)
#     derivative = numerator/(12*h)
#     return derivative



# # use this template if you have a
# # simple monomial such as cos, sin, log, ln, e^x, etc.
# # np.sin is the function handle for sin
# g_1 = np.sin
# x_1 = 2*np.pi # 2*pi
# h_step1 = .01 # step size/h value
# # The line below calls function we created earlier
# approx_1 = five_step(g_1, x_1, h_step1)

# print(f"The approximate derivative of your 1st function \n"
#       f" at x_0 = {x_1} is: {approx_1}")



# # Adding a line break for visual purposes
# print(f"\n\n\n")
# #%%
# # This is an extra part to show that the five-point method works for other functions as well
# # use this template if you have a complicated
# # monomial, or polynomial; for example x^2, e^2x
# def g_2(x):
#     # this is an example using x^4 + x^2 + 1
#     return x**4 + x**2 + 1
# x_2 = 1
# h_step2 = .01
# approx_2 = five_step(g_2, x_2, h_step2)

# print(f"The approximate derivative of your 2nd function \n"
#       f" at x_0 = {x_2} is: {approx_2}")
#%%
import numpy as np
import matplotlib.pyplot as plt

# For our function we will be using cos(x) -> d/dx(cos(x)) = -sin(x)
# Below we import the function we created from Task 1

# Our Five-Point Formula Function
def five_step(f, x_0, h):
    numerator = f(x_0 - 2*h) - 8*f(x_0 - h) + 8*f(x_0 + h) - f(x_0 + 2*h)
    derivative = numerator/(12*h)
    return derivative

def g(x):
    return np.sqrt(1 - x**2) # cos(x)
x1 = 0.005 # pi/4
#%%
# This is the exact value for d/dx(cos(x))
# evaluated at pi/4
exact = -x1 / np.sqrt(1-x1**2)
h_vec = [] # Vector to store h values
approx_vec = [] # Vector to store approximate values using Formula
error = [] # Vector to store absolute error
error_ratio =[] # Vector to store ratio's between i and i-1 error term

# We will use this to create our series of decreasing h values
for j in range(1,15):
    k = 1/(2**j)
    h_vec.append(k)

# (I.) This will be used to compute the value of -sin(x) using each h value
for i in h_vec:
    approx = five_step(g, x1, i)
    approx_vec.append(approx)

print("The approximate values at each h value are:")
for z in approx_vec:
    print(f"{z}\n")
    #%%
# (II.) This will be used to acquire the absolute error at each step size
for t in approx_vec:
    e_value = abs(exact - t)
    error.append(e_value)

print("The absolute error for your five-point method of each respective h size is: ")

# prints error[] vector
for i in error:
    print(f"{i}\n")

print(f"\n\n")
#%%
# (III.) This will be used to compute the ratio between i and i-1 term from error_vec
for i in range(0, len(error)):
    e_ratio = error[i]/error[(i-1)]
    error_ratio.append(e_ratio)

print("The error ratio between the i and i-1 error term is:")

# prints error_ratio[] vector
for i in error_ratio:
    print(f"{i}\n")
#%%
# This was used to create the line of slope 4 in our loglog plot
# The reason why we got to this conclusion is discussed in the results report
y = [] # Used to store the respective y-values for our line of slope 4
for n in h_vec:
    y_val = n**4
    y.append(y_val)


# Part B -----------------------------------------------------------------

# Plotting our approximation error results onto a loglog plot

plt.figure(figsize = (12, 8))
plt.loglog(h_vec, error, label="Error in Log")
plt.loglog(h_vec, y, label="Slope 4")
plt.xlabel('h')
plt.ylabel('Numerical Error')
plt.title("Numerical Error vs. h Size")
plt.legend()
plt.grid(True)
plt.show()

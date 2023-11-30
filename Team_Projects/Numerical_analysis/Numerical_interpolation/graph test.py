# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 10:39:45 2023

@author: joonc
"""

import matplotlib.pyplot as plt
import numpy as np

# Define the function
# def my_function(x):
#           return 118.515 * x**0 + 5.276 * x**1 - 0.244 * x**2 + 0.003*x**3 + -0.000

def my_function(x):
   return 123.740 * x**0 + 3.475 * x**1 - 0.140 * x**2 + 0.001 * x**3

# Generate x values from 0 to 90
x_values = np.linspace(0, 100, 1000)

# Calculate corresponding y values using the function
y_values = my_function(x_values)

# Plot the graph
plt.plot(x_values, y_values, label='b(x)')
plt.title('Graph of the polynomial')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
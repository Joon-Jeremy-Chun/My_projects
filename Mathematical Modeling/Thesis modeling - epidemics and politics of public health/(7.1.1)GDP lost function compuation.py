# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 19:08:00 2025

@author: joonc
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# Define the exponential GDP loss function
def gdp_loss(x, a, b, c):
    return a * np.exp(b * x) + c

# Given data points for interpolation
x_data = np.array([0.7, 0.35, 0.2])
y_data = np.array([-0.002, -0.018, -0.064])

# Provide an initial guess for the parameters [a, b, c]
initial_guess = [-1, -1, -1]

# Increase the maximum number of function evaluations
popt, pcov = curve_fit(gdp_loss, x_data, y_data, p0=initial_guess, maxfev=2000)
a, b, c = popt

print("Fitted parameters:")
print("a = {:.4f}, b = {:.4f}, c = {:.4f}".format(a, b, c))

# Print the function equation
print("GDP_lost(beta_m) = {:.4f} * exp({:.4f} * beta_m) + {:.4f}".format(a, b, c))

# Generate a smooth set of x values for plotting the fitted curve
x_fit = np.linspace(min(x_data) - 0.1, max(x_data) + 0.1, 100)
y_fit = gdp_loss(x_fit, a, b, c)

# Plot the data points and the fitted exponential function
plt.figure(figsize=(8, 5))
plt.scatter(x_data, y_data, color='red', label='Data points')
plt.plot(x_fit, y_fit, label='Exponential fit: a·exp(b·x)+c', color='blue')
plt.xlabel('X (e.g., time factor)')
plt.ylabel('GDP Loss')
plt.title('GDP Loss Function via Exponential Interpolation')
plt.legend()
plt.grid(True)

# Create the 'Figures' directory if it doesn't exist and save the figure
os.makedirs('Figures', exist_ok=True)
plt.savefig("Figures/GDP_loss_function.png")

plt.show()




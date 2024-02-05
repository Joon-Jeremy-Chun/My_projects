# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 17:10:26 2024

@author: joonc
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import integrate

recursive_calls_counter = 0  # Counter to track recursive calls

def f(x):
    return np.sin(x)  # Example function, replace it with your own function

def trapezoidal_rule(a, b, f_a, f_b):
    return (b - a) * (f_a + f_b) / 2

def simpsons_rule(a, b, f_a, f_b, c, f_c):
    return (b - a) * (f_a + 4 * f_c + f_b) / 6

def adaptive_quadrature(a, b, f_a, f_b, tolerance):
    global recursive_calls_counter  # Access the global counter variable
    recursive_calls_counter += 1

    c = (a + b) / 2
    f_c = f(c)

    trapezoidal_approx = trapezoidal_rule(a, b, f_a, f_b)
    simpsons_approx = simpsons_rule(a, b, f_a, f_b, c, f_c)
    
    error = abs(simpsons_approx - trapezoidal_approx)

    if error < tolerance:
        return simpsons_approx, error
    else:
        left_half, error_left = adaptive_quadrature(a, c, f_a, f_c, tolerance / 2)
        right_half, error_right = adaptive_quadrature(c, b, f_c, f_b, tolerance / 2)
        total_error = error_left + error_right
        return left_half + right_half, total_error

# Parameters
a = 0
b = np.pi
tolerance = 1e-6

# Compute the exact value of the test function
exact_value, error_exact_value = integrate.quad(f, a, b)

# Define an array of the tolerance values
tolerance_values = [2**(-i) for i in range(1, 20)]

# Arrays to store results, recursive call counts, absolute errors, and error ratios
results = []
recursive_calls_counts = []
absolute_errors = []
error_ratios = []

# Compute the results and errors
for tol in tolerance_values:
    recursive_calls_counter = 0  # Reset the global counter
    result, total_error = adaptive_quadrature(a, b, f(a), f(b), tol)
    absolute_error = abs(exact_value - result)
    error_ratio = total_error / tol

    print(f"For tolerance = {tol}, approx = {result}, absolute_error = {absolute_error}, error_ratio = {error_ratio}, recursive_calls = {recursive_calls_counter}")
    
    results.append(result)
    recursive_calls_counts.append(recursive_calls_counter)
    absolute_errors.append(absolute_error)
    error_ratios.append(error_ratio)

# Plotting
plt.figure(figsize=(12, 8))

# Plot Number of Recursive Calls vs Tolerance
plt.subplot(2, 2, 1)
plt.plot(tolerance_values, recursive_calls_counts, 'o-', label='Recursive Calls')
plt.title('Number of Recursive Calls vs Tolerance')
plt.xlabel('Tolerance')
plt.ylabel('Recursive Calls')
plt.xscale('log')  # Use a logarithmic scale for better visualization
plt.legend()
plt.grid(True)

# Plot Absolute Errors vs Tolerance
plt.subplot(2, 2, 2)
plt.plot(tolerance_values, absolute_errors, 'o-', label='Absolute Errors')
plt.title('Absolute Errors vs Tolerance')
plt.xlabel('Tolerance')
plt.ylabel('Absolute Errors')
plt.xscale('log')  # Use a logarithmic scale for better visualization
plt.legend()
plt.grid(True)

# Plot Error Ratios vs Tolerance
plt.subplot(2, 2, 3)
plt.plot(tolerance_values, error_ratios, 'o-', label='Error Ratios')
plt.title('Error Ratios vs Tolerance')
plt.xlabel('Tolerance')
plt.ylabel('Error Ratios')
plt.xscale('log')  # Use a logarithmic scale for better visualization
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

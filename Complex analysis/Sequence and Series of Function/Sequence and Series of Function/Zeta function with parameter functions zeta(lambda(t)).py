# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 18:30:55 2025

@author: joonc
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_complex_inputs(t_vals, x_func, y_func):
    """
    Generates an array of complex numbers s = x(t) + i*y(t)
    for a given array of parameter values t, using the provided
    functions x_func and y_func.

    Parameters:
        t_vals (array-like): Array of parameter values.
        x_func (function): Function of t that returns the real part x(t).
        y_func (function): Function of t that returns the imaginary part y(t).

    Returns:
        np.array: Array of complex numbers s.
    """
    return np.array([x_func(t) + 1j * y_func(t) for t in t_vals])

def zeta(s, terms=100000):
    """
    Approximates the Riemann zeta function for a complex s (with Re(s) > 1)
    using a finite series.

    Parameters:
        s (complex): The complex number where the function is evaluated.
        terms (int): The number of terms in the series approximation.

    Returns:
        complex: The approximated value of the zeta function.
    """
    sum_value = 0 + 0j
    for n in range(1, terms + 1):
        sum_value += 1 / (n ** s)
    return sum_value

def plot_results(s_vals, zeta_vals):
    """
    Plots the input complex numbers and corresponding zeta outputs on two
    side-by-side subplots. Only continuous lines are displayed (no markers),
    with axis limits set to -5 to 5 for both x and y.

    Parameters:
        s_vals (np.array): Array of input complex numbers.
        zeta_vals (list): List of approximated zeta(s) values corresponding to s_vals.
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left subplot: Plot input s(t) as a continuous blue line.
    ax_left.plot(s_vals.real, s_vals.imag, 'b-', label='Input s(t)')
    ax_left.set_xlabel('Real Part')
    ax_left.set_ylabel('Imaginary Part')
    ax_left.set_title('Input Complex Numbers: s(t)')
    ax_left.grid(True)
    ax_left.legend()
    ax_left.set_xlim(-5, 5)
    ax_left.set_ylim(-5, 5)
    
    # Right subplot: Plot output zeta(s) as a continuous red line.
    ax_right.plot([z.real for z in zeta_vals], [z.imag for z in zeta_vals],
                  'r-', label='Output zeta(s)')
    ax_right.set_xlabel('Real Part')
    ax_right.set_ylabel('Imaginary Part')
    ax_right.set_title('Output Complex Numbers: zeta(s)')
    ax_right.grid(True)
    ax_right.legend()
    ax_right.set_xlim(-5, 5)
    ax_right.set_ylim(-5, 5)
    
    plt.tight_layout()
    plt.show()

def main():
    # Define 100 equally spaced t values between ....
    t_vals = np.linspace(0, 3, 100)
    
    # Parameter equations for the input complex numbers.
    x_func = lambda t: 2 + 3*t
    y_func = lambda t: 2 + 1*t
    
    # Generate input complex numbers using the parameter functions.
    s_vals = generate_complex_inputs(t_vals, x_func, y_func)
    
    # Compute zeta(s) for each input s (using 100,000 terms for better accuracy).
    zeta_vals = [zeta(s, terms=100000) for s in s_vals]
    
    # Plot the entire path of s(t) and zeta(s(t)).
    plot_results(s_vals, zeta_vals)
    
    # Print the zeta values for the FIRST and LAST points in s_vals.
    s_first = s_vals[0]
    zeta_first = zeta_vals[0]
    print(f"First value of s in the array is: {s_first}")
    print(f"zeta(s_first) with 100000 terms ≈ {zeta_first}")
    
    s_last = s_vals[-1]
    zeta_last = zeta_vals[-1]
    print(f"Last value of s in the array is: {s_last}")
    print(f"zeta(s_last) with 100000 terms ≈ {zeta_last}")

if __name__ == "__main__":
    main()

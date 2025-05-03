# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 19:00:29 2025

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

def plot_results_multi(s_vals_list, zeta_vals_list, labels, colors_input, colors_output):
    """
    Plots multiple sets of input complex numbers and corresponding zeta outputs
    on two side-by-side subplots. Each set is drawn with its specified color.
    
    Parameters:
        s_vals_list (list of np.array): List of arrays of input complex numbers.
        zeta_vals_list (list of list): List of zeta(s) arrays corresponding to s_vals.
        labels (list of str): List of labels for each parameter set.
        colors_input (list of str): Colors to use for input curves.
        colors_output (list of str): Colors to use for output curves.
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left subplot: Plot each set of input s(t) as a continuous line.
    for s_vals, label, color in zip(s_vals_list, labels, colors_input):
        ax_left.plot(s_vals.real, s_vals.imag, '-', color=color, label=label)
    ax_left.set_xlabel('Real Part')
    ax_left.set_ylabel('Imaginary Part')
    ax_left.set_title('Input Complex Numbers: s(t)')
    ax_left.grid(True)
    ax_left.legend()
    ax_left.set_xlim(-5, 5)
    ax_left.set_ylim(-5, 5)
    
    # Right subplot: Plot each set of output zeta(s) as a continuous line.
    for z_vals, label, color in zip(zeta_vals_list, labels, colors_output):
        ax_right.plot([z.real for z in z_vals], [z.imag for z in z_vals],
                      '-', color=color, label=label)
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
    # Define 100 equally spaced t values between ...
    t_vals = np.linspace(0, 5, 100)
    
    # Define two different parameter equations.
    # Parameter Equation 1:
    # x_func1 = lambda t: 2
    # y_func1 = lambda t: 0.5 + t
    
    x_func1 = lambda t:   2
    y_func1 = lambda t:   2 +t
    
    # Parameter Equation 2:
    x_func2 = lambda t: 2 + t
    y_func2 = lambda t: 2
    
    # Generate input complex numbers for both equations.
    s_vals1 = generate_complex_inputs(t_vals, x_func1, y_func1)
    s_vals2 = generate_complex_inputs(t_vals, x_func2, y_func2)
    
    # Compute zeta(s) for each input s (using 100,000 terms).
    zeta_vals1 = [zeta(s, terms=100000) for s in s_vals1]
    zeta_vals2 = [zeta(s, terms=100000) for s in s_vals2]
    
    # Prepare lists for plotting.
    s_vals_list = [s_vals1, s_vals2]
    zeta_vals_list = [zeta_vals1, zeta_vals2]
    labels = ['Param Eq 1:', 'Param Eq 2:']
    colors_input = ['blue', 'green']   # Colors for input curves.
    colors_output = ['red', 'magenta']   # Colors for output curves.
    
    # Plot the results.
    plot_results_multi(s_vals_list, zeta_vals_list, labels, colors_input, colors_output)
    
    # Optionally, print the first and last values for both parameter sets.
    for i, (s_vals, z_vals, label) in enumerate(zip(s_vals_list, zeta_vals_list, labels)):
        print(f"\nFor {label}:")
        print(f"  First value of s is: {s_vals[0]}, zeta(s) ≈ {z_vals[0]}")
        print(f"  Last value of s is: {s_vals[-1]}, zeta(s) ≈ {z_vals[-1]}")

if __name__ == "__main__":
    main()

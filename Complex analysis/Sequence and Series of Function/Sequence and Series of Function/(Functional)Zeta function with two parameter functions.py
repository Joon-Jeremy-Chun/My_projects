# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 20:57:20 2025

@author: joonc
"""
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt

def generate_complex_inputs(t_vals, x_func, y_func):
    """
    Generates an array of complex numbers s = x(t) + i*y(t)
    for the given parameter values t.
    """
    return np.array([x_func(t) + 1j * y_func(t) for t in t_vals], dtype=complex)

def compute_zeta_for_inputs(s_vals, terms=100000):
    """
    Computes zeta(s) for each s in s_vals using mpmath's zeta function.
    """
    zeta_vals = []
    for s in s_vals:
        # Convert Python complex to mpmath's mpc type.
        s_mpc = mp.mpc(s.real, s.imag)
        zeta_vals.append(mp.zeta(s_mpc))
    return zeta_vals

def plot_results_multi(s_vals_list, zeta_vals_list, labels, colors_input, colors_output):
    """
    Plots multiple sets of input complex numbers and their corresponding zeta(s)
    outputs on side-by-side subplots.
    
    Parameters:
      - s_vals_list: List of arrays of input complex numbers.
      - zeta_vals_list: List of corresponding lists of zeta(s) values.
      - labels: List of labels for each parameterized line.
      - colors_input: List of colors for the input lines.
      - colors_output: List of colors for the output lines.
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left subplot: Plot the input lines.
    for s_vals, label, color in zip(s_vals_list, labels, colors_input):
        ax_left.plot(s_vals.real, s_vals.imag, '-', color=color, label=label)
    ax_left.set_xlabel("Real Part")
    ax_left.set_ylabel("Imaginary Part")
    ax_left.set_title("Input Complex Numbers: s(t)")
    ax_left.grid(True)
    ax_left.legend()
    ax_left.set_xlim(-5, 5)
    ax_left.set_ylim(-5, 5)
    
    # Right subplot: Plot the corresponding output zeta(s) lines.
    for z_vals, label, color in zip(zeta_vals_list, labels, colors_output):
        ax_right.plot([float(z.real) for z in z_vals], [float(z.imag) for z in z_vals],
                      '-', color=color, label=label)
    ax_right.set_xlabel("Real Part")
    ax_right.set_ylabel("Imaginary Part")
    ax_right.set_title("Output: zeta(s) in the Complex Plane")
    ax_right.grid(True)
    ax_right.legend()
    ax_right.set_xlim(-5, 5)
    ax_right.set_ylim(-5, 5)
    
    plt.tight_layout()
    plt.show()

# ---------------------------
# Main Code
# ---------------------------
# Define the parameter range
t_vals = np.linspace(1, 5, 1000)

# Define two different parameter equations.
# Line 1: 
# x_func1 = lambda t: 1 + t
# y_func1 = lambda t: 1

# x_func1 = lambda t: 1.5
# y_func1 = lambda t: -1 + t

x_func1 = lambda t: 1.1 + t
y_func1 = lambda t: 1.1 + t

# x_func1 = lambda t: 1/2
# y_func1 = lambda t: t

# Line 2:
# x_func2 = lambda t: 1 + t
# y_func2 = lambda t: -1

x_func2 = lambda t: 1.1 + t
y_func2 = lambda t: 1.1

# Generate input complex numbers for both lines.
s_vals1 = generate_complex_inputs(t_vals, x_func1, y_func1)
s_vals2 = generate_complex_inputs(t_vals, x_func2, y_func2)

# Compute zeta(s) for each input s (using 100,000 terms for better accuracy)
zeta_vals1 = compute_zeta_for_inputs(s_vals1, terms=100000)
zeta_vals2 = compute_zeta_for_inputs(s_vals2, terms=100000)

# Prepare lists for plotting.
s_vals_list = [s_vals1, s_vals2]
zeta_vals_list = [zeta_vals1, zeta_vals2]
labels = ["Line 1:", "Line 2:"]
colors_input = ["blue", "green"]       # Colors for the input lines.
colors_output = ["red", "magenta"]       # Colors for the output lines.

# Plot the two lines on the same plot.
plot_results_multi(s_vals_list, zeta_vals_list, labels, colors_input, colors_output)

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

def zeta_functional_equation_rhs(s):
    """
    Computes:
      2^s * pi^(s-1) * sin(pi*s/2) * Gamma(1-s) * zeta(1-s)
    using mpmath functions.
    Expects `s` as an mpmath mpc.
    """
    sin   = mp.sin
    gamma = mp.gamma
    pi    = mp.pi
    zeta  = mp.zeta
    return 2**s * pi**(s - 1) * sin(pi * s / 2) * gamma(1 - s) * zeta(1 - s)

def plot_results_multi(s_vals_list, zeta_vals_list, labels, colors_input, colors_output):
    """
    Plots multiple sets of input complex numbers and their corresponding zeta(s)
    outputs on side-by-side subplots.
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left: input curves
    for s_vals, label, color in zip(s_vals_list, labels, colors_input):
        ax_left.plot(s_vals.real, s_vals.imag, '-', color=color, label=label)
    ax_left.set_xlabel("Real Part")
    ax_left.set_ylabel("Imaginary Part")
    ax_left.set_title("Input Complex Numbers: s(t)")
    ax_left.grid(True)
    ax_left.legend()
    ax_left.set_xlim(-5, 5)
    ax_left.set_ylim(-5, 5)
    
    # Right: zeta(s) curves
    for z_vals, label, color in zip(zeta_vals_list, labels, colors_output):
        ax_right.plot([float(z.real) for z in z_vals],
                      [float(z.imag) for z in z_vals],
                      '-', color=color, label=label)
    ax_right.set_xlabel("Real Part")
    ax_right.set_ylabel("Imaginary Part")
    ax_right.set_title("Output: ζ(s) in the Complex Plane")
    ax_right.grid(True)
    ax_right.legend()
    ax_right.set_xlim(-5, 5)
    ax_right.set_ylim(-5, 5)
    
    plt.tight_layout()
    plt.show()

# -----------------------------------
# Main
# -----------------------------------
if __name__ == "__main__":
    # Parameter range
    t_vals = np.linspace(0, 50, 1000)
    
    # # Two parameterizations of the line Re(s)=1/2
    # x_func1 = lambda t: 1/2
    # y_func1 = lambda t:  t
    # x_func2 = lambda t: 1/2
    # y_func2 = lambda t: -t
    
    
    # Two parameterizations of the line Re(s)=1/4
    x_func1 = lambda t: 1/4
    y_func1 = lambda t:  t
    x_func2 = lambda t: 1/4
    y_func2 = lambda t: -t
    
    # Generate s(t) arrays
    s_vals1 = generate_complex_inputs(t_vals, x_func1, y_func1)
    s_vals2 = generate_complex_inputs(t_vals, x_func2, y_func2)
    
    # Compute ζ(s) via the functional equation, element-wise
    zeta_vals1 = [
        zeta_functional_equation_rhs(mp.mpc(s.real, s.imag))
        for s in s_vals1
    ]
    zeta_vals2 = [
        zeta_functional_equation_rhs(mp.mpc(s.real, s.imag))
        for s in s_vals2
    ]
    
    # Plot
    s_vals_list    = [s_vals1, s_vals2]
    zeta_vals_list = [zeta_vals1, zeta_vals2]
    labels         = ["Line 1: Im≥0", "Line 2: Im≤0"]
    colors_input   = ["blue", "green"]
    colors_output  = ["red", "magenta"]
    
    plot_results_multi(s_vals_list, zeta_vals_list,
                       labels, colors_input, colors_output)

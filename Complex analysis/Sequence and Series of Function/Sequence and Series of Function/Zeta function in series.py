# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 13:03:29 2025

@author: joonc
"""
import matplotlib.pyplot as plt

def zeta(s, terms=1000000):
    """
    Approximates the Riemann zeta function for a complex s (with Re(s) > 1)
    using a finite series.

    Parameters:
        s (complex): The complex number where the function is evaluated.
        terms (int): The number of terms in the series approximation.

    Returns:
        complex: The approximated value of the zeta function.
    """
    sum_value = 0 + 0j  # Initialize as a complex number
    for n in range(1, terms + 1):
        sum_value += 1 / (n ** s)
    return sum_value

def plot_results(s, result):
    """
    Plots the input complex number s and the output zeta(s)
    on two side-by-side subplots with axis limits from -10 to 10.
    
    Left subplot: input complex plane.
    Right subplot: output complex plane.
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))

    # Left subplot: Plot the input complex number s
    ax_left.scatter(s.real, s.imag, color='blue', label='s = {:.2f} {:+.2f}i'.format(s.real, s.imag))
    ax_left.axhline(0, color='black', lw=0.5)
    ax_left.axvline(0, color='black', lw=0.5)
    ax_left.set_xlim(-10, 10)
    ax_left.set_ylim(-10, 10)
    ax_left.set_xlabel('Real Part')
    ax_left.set_ylabel('Imaginary Part')
    ax_left.set_title('Input Complex Plane')
    ax_left.legend()
    ax_left.grid(True)

    # Right subplot: Plot the output complex number zeta(s)
    ax_right.scatter(result.real, result.imag, color='red', label='zeta(s) = {:.5f} {:+.5f}i'.format(result.real, result.imag))
    ax_right.axhline(0, color='black', lw=0.5)
    ax_right.axvline(0, color='black', lw=0.5)
    ax_right.set_xlim(-10, 10)
    ax_right.set_ylim(-10, 10)
    ax_right.set_xlabel('Real Part')
    ax_right.set_ylabel('Imaginary Part')
    ax_right.set_title('Output Complex Plane')
    ax_right.legend()
    ax_right.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    s = 2 + -1j  # Complex number input (x + yi)
    result = zeta(s)
    print("zeta({}) approximated using {} terms is:".format(s, 1000000))
    print(result)
    plot_results(s, result)

if __name__ == "__main__":
    main()




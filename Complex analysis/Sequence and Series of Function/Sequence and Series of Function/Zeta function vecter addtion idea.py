# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:55:04 2025

@author: joonc
"""
import numpy as np
import matplotlib.pyplot as plt

def partial_sums_zeta(s, max_terms=100000):

    partial_sums = []
    current_sum = 0 + 0j
    for n in range(1, max_terms + 1):
        current_sum += 1 / (n ** s)
        partial_sums.append(current_sum)
    return partial_sums

def plot_input_and_vectors(s, partial_sums):

    # Final approximate zeta(s) is the last partial sum
    final_sum = partial_sums[-1]

    # Prepare figure with two subplots
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))

    # -------------------------------------------------------------------------
    # Left Subplot: Input Complex Plane
    # -------------------------------------------------------------------------
    ax_left.scatter(s.real, s.imag, color='blue', s=100,
                    label=f"s = {s.real:.2f} {s.imag:+.2f}i")
    ax_left.axhline(0, color='black', lw=0.5)
    ax_left.axvline(0, color='black', lw=0.5)
    ax_left.set_xlim(-10, 10)
    ax_left.set_ylim(-10, 10)
    ax_left.set_xlabel("Real Part")
    ax_left.set_ylabel("Imaginary Part")
    ax_left.set_title("Input Complex Plane")
    ax_left.legend()
    ax_left.grid(True)

    # -------------------------------------------------------------------------
    # Right Subplot: Polyline of Partial Sums
    # -------------------------------------------------------------------------
    # Insert the origin at the start for convenience
    cumulative = np.insert(partial_sums, 0, 0+0j)

    # Extract real and imaginary parts for a continuous line plot
    real_vals = [z.real for z in cumulative]
    imag_vals = [z.imag for z in cumulative]

    # Plot a continuous line through the partial sums
    ax_right.plot(real_vals, imag_vals, color='blue', label='Partial Sums Path')

    # Highlight the final sum with a red dot
    ax_right.scatter(final_sum.real, final_sum.imag, color='red', s=100,
                     label=f"zeta(s) approx (N={len(partial_sums)})")

    # Axis lines, labels, etc.
    ax_right.axhline(0, color='black', lw=0.5)
    ax_right.axvline(0, color='black', lw=0.5)
    ax_right.set_xlabel("Real Part")
    ax_right.set_ylabel("Imaginary Part")
    ax_right.set_title("Vector Addition of Partial Sums")
    ax_right.legend()
    ax_right.grid(True)

    # Adjust the viewing window so we can see everything clearly
    all_re = real_vals
    all_im = imag_vals
    pad = 0.5
    ax_right.set_xlim(min(all_re) - pad, max(all_re) + pad)
    ax_right.set_ylim(min(all_im) - pad, max(all_im) + pad)

    plt.tight_layout()
    plt.show()

def main():
    # Example: s = x + iy (with Re(s) > 1 for a decent approximation)
    s = 2  + 1j

    # Number of terms to sum (and show in the diagram)
    max_terms = 1000000

    # Compute partial sums
    partial_sums = partial_sums_zeta(s, max_terms)

    # Print the final approximate value
    print(f"Using {max_terms} terms to approximate zeta({s}):")
    print(f"zeta({s}) ≈ {partial_sums[-1]}")

    # Plot side-by-side: input plane (left), partial-sums path (right)
    plot_input_and_vectors(s, partial_sums)

if __name__ == "__main__":
    main()



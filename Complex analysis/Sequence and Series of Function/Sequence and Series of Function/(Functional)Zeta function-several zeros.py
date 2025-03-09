# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 02:50:48 2025

@author: joonc
"""
import mpmath as mp
import matplotlib.pyplot as plt

def zeta_functional_equation_rhs(s):
    """
    Computes:
      2^s * pi^(s-1) * sin(pi*s/2) * Gamma(1-s) * zeta(1-s)
    using mpmath functions.
    """
    sin = mp.sin
    gamma = mp.gamma
    pi = mp.pi
    zeta = mp.zeta
    return (2**s * pi**(s - 1) * sin(pi*s/2) * gamma(1 - s) * zeta(1 - s))

# ------------------------------
# Set precision and define input zeros
# ------------------------------
mp.mp.prec = 80  # set precision

# List of 10 known positive imaginary parts for zeros on the critical line
initial_guesses = [14.134725, 21.022040, 25.010857, 30.424876,
                   32.935062, 37.586178, 40.918719, 43.327073,
                   48.005151, 49.773832]

# Create zeros on the critical line: 1/2 + i*(gamma) and 1/2 - i*(gamma)
zeros_pos = [mp.mpc('1/2', str(gamma)) for gamma in initial_guesses]
zeros_neg = [mp.conj(z) for z in zeros_pos]
zeros = zeros_pos + zeros_neg  # Total 20 zeros

# ------------------------------
# Compute zeta(s) for each zero using direct method and functional equation
# ------------------------------
zeta_direct_values = [mp.zeta(z) for z in zeros]
zeta_fe_values = [zeta_functional_equation_rhs(z) for z in zeros]

print("Displaying 20 zeros (10 with positive imaginary part and 10 with negative imaginary part):\n")
for z, zd, zfe in zip(zeros, zeta_direct_values, zeta_fe_values):
    print("Input s:               ", z)
    print("Direct zeta(s):         ", zd)
    print("Functional eqn zeta(s): ", zfe)
    print("Difference:             ", abs(zd - zfe))
    print("--------------------------------------------------")

# ------------------------------
# Plotting: Input zeros and computed zeta(s) outputs in the complex plane
# ------------------------------
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))

# Left subplot: Plot the input zeros (s values)
ax_left.scatter([float(z.real) for z in zeros],
                [float(z.imag) for z in zeros],
                color='blue',
                label="s = 1/2 ± iγ")
ax_left.axhline(0, color='black', lw=0.5)
ax_left.axvline(0, color='black', lw=0.5)
ax_left.set_xlim(-1, 2)
# Adjust y-limit to accommodate zeros (approx. -50 to 50)
ax_left.set_ylim(-55, 55)
ax_left.set_xlabel("Real Part")
ax_left.set_ylabel("Imaginary Part")
ax_left.set_title("Input: Zeros on the Critical Line")
ax_left.legend()
ax_left.grid(True)

# Right subplot: Plot the output zeta(s) computed directly
ax_right.scatter([float(z.real) for z in zeta_direct_values],
                 [float(z.imag) for z in zeta_direct_values],
                 color='red',
                 label="zeta(s) (Direct)")
ax_right.axhline(0, color='black', lw=0.5)
ax_right.axvline(0, color='black', lw=0.5)
ax_right.set_xlim(-10, 10)
ax_right.set_ylim(-10, 10)
ax_right.set_xlabel("Real Part")
ax_right.set_ylabel("Imaginary Part")
ax_right.set_title("Output: zeta(s) in the Complex Plane")
ax_right.legend()
ax_right.grid(True)

plt.tight_layout()
plt.show()



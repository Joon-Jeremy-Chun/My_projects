# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 20:30:29 2025

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
# Set precision and define input
# ------------------------------
mp.mp.prec = 80  # set precision
# s = mp.mpc('-1', '0')  # Complex number input (x + yi)

s = mp.mpc('1/2', '14.134725')  # Complex number input (x + yi)
initial_guesses = [14.134725, 21.022040, 25.010857, 30.424876,
                   32.935062, 37.586178, 40.918719, 43.327073,
                   48.005151, 49.773832]

# ------------------------------
# Compute zeta(s) both ways
# ------------------------------
zeta_direct = mp.zeta(s)
zeta_fe = zeta_functional_equation_rhs(s)

print("Input s:", s)
print("Direct zeta(s):         ", zeta_direct)
print("Functional eqn zeta(s): ", zeta_fe)
print("Difference:             ", abs(zeta_direct - zeta_fe))

# ------------------------------
# Plotting
# ------------------------------
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))

# Left subplot: Plot the input s.
ax_left.scatter(float(s.real), float(s.imag), color='blue',
                label=f"s = {float(s.real):.2f} {float(s.imag):+0.2f}i")
ax_left.axhline(0, color='black', lw=0.5)
ax_left.axvline(0, color='black', lw=0.5)
ax_left.set_xlim(-10, 10)
ax_left.set_ylim(-10, 10)
ax_left.set_xlabel("Real Part")
ax_left.set_ylabel("Imaginary Part")
ax_left.set_title("Input: s in the Complex Plane")
ax_left.legend()
ax_left.grid(True)

# Right subplot: Plot the output zeta(s) (using direct computation).
ax_right.scatter(float(zeta_direct.real), float(zeta_direct.imag), color='red',
                 label=f"zeta(s) = {float(zeta_direct.real):.5f} {float(zeta_direct.imag):+0.5f}i")
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







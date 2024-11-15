# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 14:53:48 2024

@author: joonc
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Question:
# Plot the domain D of the function f(z) = ln(z^2 - 2) with the branch cut
# defined such that Arg(z^2 - 2) ∈ (π/2, 5π/2). Highlight the excluded regions
# (branch cuts and singularities) in the complex plane.
# -----------------------------------------------------------------------------

# Define the range for x and y axes
x_min, x_max, x_points = -4, 4, 1000
y_min, y_max, y_points = -4, 4, 1000

# Create a grid of x and y values
x = np.linspace(x_min, x_max, x_points)
y = np.linspace(y_min, y_max, y_points)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y  # Complex grid

# Compute z^2 - 2
w = Z**2 - 2

# Compute the argument of w
arg_w = np.angle(w)

# Set the branch cut boundaries
arg_min = np.pi / 2
arg_max = 5 * np.pi / 2

# Adjust arguments to be between 0 and 2*pi
arg_w_mod = np.mod(arg_w, 2 * np.pi)

# Determine the excluded points (branch cuts and singularities)
# Exclude points where arg(w) equals the branch cut boundaries
tolerance = 1e-3  # Adjust tolerance if necessary
on_branch_cut = np.abs(arg_w_mod - arg_min) < tolerance

# Exclude the singularities where w = 0
singularity = np.abs(w) < 1e-8

# Combine the excluded points
excluded_points = on_branch_cut | singularity

# Prepare the plot
plt.figure(figsize=(8, 8))

# Plot the domain D where f(z) is defined
# Included points in light blue
plt.pcolormesh(X, Y, ~excluded_points, shading='auto', cmap='Blues', alpha=0.7)

# Highlight the excluded points in red with increased linewidth
# Use contour to make the excluded regions more visible
plt.contour(X, Y, excluded_points.astype(int), levels=[0.5], colors='red', linewidths=2)

# Mark the singularities with larger black dots
plt.scatter([np.sqrt(2), -np.sqrt(2)], [0, 0], color='black', s=100, zorder=5, label='Singularities')

# Customize the plot
plt.title('Domain D of $f(z) = \\ln(z^2 - 2)$ with Excluded Regions Highlighted')
plt.xlabel('Real Part (x)')
plt.ylabel('Imaginary Part (y)')
plt.axis('equal')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Add legend for singularities
plt.legend()

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()





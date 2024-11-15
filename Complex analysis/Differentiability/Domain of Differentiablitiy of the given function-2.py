# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 15:11:02 2024

@author: joonc
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
#not completed code
# -----------------------------------------------------------------------------
# Question:
# Plot the domain D of the function f(z) = ln(sin(z)) with the branch cut
# defined such that Arg(sin(z)) ∈ (-π, π). Highlight all the vertical branch cuts
# and singularities in the complex plane.
# -----------------------------------------------------------------------------

# Define the range for x and y axes
x_min, x_max, x_points = -20 * np.pi, 20 * np.pi, 4000
y_min, y_max, y_points = -4, 4, 2000

# Create a grid of x and y values
x = np.linspace(x_min, x_max, x_points)
y = np.linspace(y_min, y_max, y_points)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y  # Complex grid

# Initialize the excluded points array
excluded_points = np.zeros_like(X, dtype=bool)

# Identify singularities at z = nπ
n_values_sing = np.arange(int(x_min / np.pi) - 1, int(x_max / np.pi) + 2)
singularity_x = n_values_sing * np.pi

# Mark singularities on the grid
for x0 in singularity_x:
    excluded_points |= np.abs(X - x0) < (np.pi / x_points)  # Small band around singularity

# Identify branch cuts along the real axis where y = 0 and sin(x) ≤ 0
y_zero = np.abs(Y) < (np.pi / y_points)
sin_x = np.sin(X)
branch_cuts_real_axis = y_zero & (sin_x <= 0)
excluded_points |= branch_cuts_real_axis

# Identify vertical branch cuts at x = (π/2) + π n where n is odd
n_start = int((x_min - np.pi / 2) / np.pi) - 1
n_end = int((x_max - np.pi / 2) / np.pi) + 2
n_values = np.arange(n_start, n_end + 1)

# Filter odd n values
n_values_odd = n_values[n_values % 2 != 0]

# Initialize branch cuts for vertical lines
branch_cuts_vertical = np.zeros_like(X, dtype=bool)

# Mark branch cuts on the grid
for n in n_values_odd:
    x_line = (np.pi / 2) + np.pi * n
    branch_cuts_vertical |= np.abs(X - x_line) < (np.pi / x_points)

excluded_points |= branch_cuts_vertical

# Prepare the plot
plt.figure(figsize=(12, 8))

# Plot the domain D where f(z) is differentiable (included points)
plt.pcolormesh(X, Y, ~excluded_points, shading='auto', cmap='Blues', alpha=0.6)

# Overlay the branch cuts as red lines
plt.contour(X, Y, excluded_points.astype(int), levels=[0.5], colors='red', linewidths=1.5)

# Mark the singularities with black dots
singularity_y = np.zeros_like(singularity_x)
plt.scatter(singularity_x, singularity_y, color='black', s=50, zorder=5, label='Singularities (z = nπ)')

# Customize the plot
plt.title('Domain \( D \) of \( f(z) = \ln(\sin(z)) \) with All Branch Cuts Highlighted')
plt.xlabel('Real Part (x)')
plt.ylabel('Imaginary Part (y)')
plt.axis('equal')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Create custom legend handles
included_patch = Patch(color='lightblue', label='Domain D (Included Points)')
branch_cut_line = Patch(color='red', label='Branch Cuts')
singularity_marker = Patch(color='black', label='Singularities (z = nπ)')

# Add legend to the plot
plt.legend(handles=[included_patch, branch_cut_line, singularity_marker], loc='upper right')

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()



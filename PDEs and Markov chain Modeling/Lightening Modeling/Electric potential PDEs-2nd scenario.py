# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:31:18 2024

@author: joonc
"""
# Use Laplace equation
# Finite Difference Method
# Central difference approximation

import numpy as np
import matplotlib.pyplot as plt

# Grid parameters
nx, ny = 100, 100  # Number of grid points
dx, dy = 1.0, 1.0  # Distance between grid points

# Create a 2D grid
phi = np.zeros((nx, ny))

# Boundary conditions (simplified model of a lightning strike)
phi[0, :] = -100  # High negative potential at the top boundary (cloud)
phi[-1, :] = 0    # Ground potential at the bottom boundary

# Set high positive potential at the point (50, 50)
phi[49, 49] = 100  # High positive potential at the specified point

# Iteration parameters
max_iter = 1000  # Maximum number of iterations
tolerance = 1e-5  # Convergence tolerance

for iteration in range(max_iter):
    phi_old = phi.copy()
    
    # Update potential values (excluding boundaries)
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # Ensure the special point's potential remains unchanged
            if i == 49 and j == 49:
                continue
            phi[i, j] = 0.25 * (phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1])
    
    # Convergence check
    error = np.max(np.abs(phi - phi_old))
    if error < tolerance:
        print(f'Converged after {iteration} iterations')
        break

# Plot the electric potential
plt.imshow(phi, extent=[0, nx*dx, 0, ny*dy], origin='upper', cmap='viridis')
plt.colorbar(label='Electric Potential (V)')
plt.title('Electric Potential in the Atmosphere During a Lightning Strike')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

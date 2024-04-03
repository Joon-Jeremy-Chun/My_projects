# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 19:12:31 2024

@author: joonc
"""

import numpy as np
import matplotlib.pyplot as plt

# Grid parameters
nx, ny = 100, 100  # Number of grid points
dx, dy = 1.0, 1.0  # Distance between grid points

# Create a 2D grid
phi = np.zeros((nx, ny))

# Boundary conditions (simplified model of a lightning strike)
phi[0, :] = 100
#phi[0, 49:51] = 100  # High potential at the top boundary (cloud)
phi[-1, :] = 0   # Ground potential at the bottom boundary

# Iteration parameters
max_iter = 1000  # Maximum number of iterations
tolerance = 1e-5  # Convergence tolerance

for iteration in range(max_iter):
    phi_old = phi.copy()
    
    # Update potential values (excluding boundaries)
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            phi[i, j] = 0.25 * (phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1])
    
    # Convergence check
    error = np.max(np.abs(phi - phi_old))
    if error < tolerance:
        print(f'Converged after {iteration} iterations')
        break

# Plot the electric potential
plt.imshow(phi, extent=[0, nx*dx, 0, ny*dy], origin='lower', cmap='viridis')
plt.colorbar(label='Electric Potential (V)')
plt.title('Electric Potential in the Atmosphere During a Lightning Strike')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
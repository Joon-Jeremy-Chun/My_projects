# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 18:10:29 2024

@author: joonc
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 10  # Length of the axon
T = 2  # Total time
dx = 0.1  # Space step
dt = 0.01  # Time step
alpha = 1  # Diffusion coefficient, simplification for the cable properties

# Derived quantities
Nx = int(L/dx)  # Number of spatial steps
Nt = int(T/dt)  # Number of time steps
x = np.linspace(0, L, Nx)  # Spatial grid
t = np.linspace(0, T, Nt)  # Temporal grid

# Initialize potential array
V = np.zeros((Nx, Nt))

# Initial condition (e.g., a localized depolarization)
V[Nx//4, 0] = 1

# Solve the PDE (simplified cable equation) using the Forward Time Central Space (FTCS) scheme
for n in range(0, Nt-1):
    for i in range(1, Nx-1):
        V[i, n+1] = V[i, n] + alpha*dt/dx**2 * (V[i+1, n] - 2*V[i, n] + V[i-1, n])

# Plotting
plt.imshow(V, extent=[0, T, 0, L], origin='lower', aspect='auto', cmap='hot')
plt.colorbar(label='Membrane Potential')
plt.xlabel('Time')
plt.ylabel('Position along axon')
plt.title('Action Potential Propagation (Simplified Model)')
plt.show()
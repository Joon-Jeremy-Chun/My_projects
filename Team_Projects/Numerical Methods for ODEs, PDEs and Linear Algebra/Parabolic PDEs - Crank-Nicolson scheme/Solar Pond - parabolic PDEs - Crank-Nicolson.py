# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 19:59:21 2024

@author: joonc
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# Grid parameters and constants
L = 1       # Length of the domain
T = 1       # Total time
Nx = 10     # Number of spatial steps
Nt = 200    # Number of time steps
C = 0.5     # Heating rate
T0 = 60     # Surface temperature at the top of the pond
dx = L / Nx
dt = T / Nt
r =  dt / (dx**2)  # The stability parameter

#%%
# Initial conditions and boundary conditions
u = np.zeros((Nx+1, Nt+1))
u[0, :] = T0   # Boundary condition at x=0, for all time steps
for i in range(1, Nx+1):
    u[i, 0] = T0 * np.exp(-2 * i * dx)  # Initial condition at t=0, function of T0*e^(-2x)

#%%
# Matrix A
A = np.zeros((Nx, Nx))
np.fill_diagonal(A, 2 + 2*r)  # Main diagonal
np.fill_diagonal(A[:-1, 1:], -r)  # Super-diagonal
np.fill_diagonal(A[1:, :-1], -r)  # Sub-diagonal
A[-1, -2] = -2*r # Given that another boundary that Assign the value for the Last row and soncond last colon 

# Matrix B
B = np.zeros((Nx, Nx))
np.fill_diagonal(B, 2 - 2*r)  # Main diagonal
np.fill_diagonal(B[:-1, 1:], r)  # Super-diagonal
np.fill_diagonal(B[1:, :-1], r)  # Sub-diagonal
B[-1, -2] = 2*r # Given that another boundary that Assign the value for the Last row and soncond last colon 

#%%
# Define the fN vector
fN = C * (L - dx * np.arange(1, Nx+1)) 
fN[0] +=  2*r*T0 #Given bounday condtion that the first entry should be added 2rT0

#%%
#Compute by Crank_Nicolson method
def crank_nicolson(u, A, B, fN, Nx, Nt):
    # Time-stepping loop
    for n in range(0, Nt):
        # Construct the right-hand side vector
        b = np.dot(B, u[1:, n]) + fN  

        # Solve for u_{i,n+1}
        u[1:, n+1] = la.solve(A, b)

    return u
#%%

u = crank_nicolson(u, A, B, fN, Nx, Nt)
#%%
# Calculate indices for the desired times
t0_index = 0  # t = 0
t05_index = int(0.5 / dt)  # t = 0.5
t1_index = int(1.0 / dt)  # t = 1

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(u[:, t0_index], label='t = 0', color='blue')
plt.plot(u[:, t05_index], label='t = 0.5', color='green')
plt.plot(u[:, t1_index], label='t = 1', color='red')

plt.title('Numerical solution of the PDE at different times')
plt.xlabel('Spatial index')
plt.ylabel('Temperature')
plt.legend()
plt.grid(True)


# Determine the temperature at the bottom at t = 1
bottom_temp_at_t1 = u[-1, t1_index]
print(f"The temperature at the bottom of the pond at t = 1 is {bottom_temp_at_t1:.4f}°C")
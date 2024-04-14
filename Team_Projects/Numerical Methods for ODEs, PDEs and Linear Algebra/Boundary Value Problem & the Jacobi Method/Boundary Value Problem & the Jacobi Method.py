# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:03:15 2024

@author: joonc
"""
import numpy as np
import matplotlib.pyplot as plt

def jacobi(A, b, u0, tol, max_iter):
    D = np.diag(np.diag(A))
    R = A - D
    D_inv = np.diag(1 / np.diag(D))
    u_approx = u0
    error = np.linalg.norm(np.dot(A, u_approx) - b, np.inf)
    num_iter = 0

    while error > tol and num_iter < max_iter:
        u_approx = np.dot(D_inv, b - np.dot(R, u_approx))
        error = np.linalg.norm(np.dot(A, u_approx) - b, np.inf)
        num_iter += 1

    return u_approx, num_iter

# Define constants p, L, q, and mesh size h
p = 7*10**(-6)
L = 6  # Beam length in feet
q = 4*10**(-7)
h = 1/2  # Mesh size

# Define the number of intervals and number of equations (n - 1 for boundary conditions)
n = int(L / h) - 1

# Construct the nxn tridiagonal matrix A
A = np.zeros((n, n))
np.fill_diagonal(A, 2 + h**2 * p)  # Main diagonal
np.fill_diagonal(A[:-1, 1:], -1)  # Super-diagonal
np.fill_diagonal(A[1:, :-1], -1)  # Sub-diagonal

# Construct the vector b
b = np.full((n, ), -h**2 * q)

# Initial guess u0
u0 = np.ones(n)
#u0 = np.array([0,0,1,0,0,1,0,0,1,0,0])
#u0 = np.array([1,2,3,4,5,6,5,4,3,2,1])
#u0 = np.array([0,0,-1,0,0,-1,0,0,-1,0,0])
#u0 = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])

# Error tolerance and maximum number of iterations
tol = 1e-8
max_iter = 1000

# Solve the system using the Jacobi method
u_approx, num_iter = jacobi(A, b, u0, tol, max_iter)

# # Plotting the approximate solution
# x = np.linspace(h, L-h, n)
# plt.plot(x, u_approx)
# plt.xlabel('Position along the beam (x)')
# plt.ylabel('Deflection')
# plt.title('Beam Deflection Due to Uniform Load')
# plt.grid(True)
# plt.show()

# Adding the boundary conditions
x = np.linspace(0, L, n+2)  # Adjusted to include 0 and L
u_complete = np.zeros(n+2)
u_complete[1:-1] = u_approx

# Plotting the approximate solution with boundary conditions
plt.plot(x, u_complete, label='Deflection')
plt.scatter([0, L], [0, 0], color='red', label='Boundary Conditions')  # Boundary conditions
plt.xlabel('Position along the beam (x)')
plt.ylabel('Deflection')
plt.title('Beam Deflection Due to Uniform Load')
plt.legend()
plt.grid(True)
plt.show()

# Find the index of the minimum deflection value
min_deflection_index = np.argmin(u_approx) 
min_deflection_value = u_approx[min_deflection_index]
min_deflection_position = x[min_deflection_index] + h # Note: Here we add h because in the code, we put boundary conditions in both ends, so index shift one unite each.

# Print the number of Jacobi iterations, minimum deflection value, and its position
print(f'Number of Jacobi iterations: {num_iter}')
print(f'Minimum deflection value: {min_deflection_value}')
print(f'Position of minimum deflection: {min_deflection_position}')
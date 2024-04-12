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

# Error tolerance and maximum number of iterations
tol = 1e-8
max_iter = 1000

# Solve the system using the Jacobi method
u_approx, num_iter = jacobi(A, b, u0, tol, max_iter)

# Plotting the approximate solution
x = np.linspace(h, L-h, n)
plt.plot(x, u_approx)
plt.xlabel('Position along the beam (x)')
plt.ylabel('Deflection')
plt.title('Beam Deflection Due to Uniform Load')
plt.grid(True)
plt.show()

print(f'Number of Jacobi iterations: {num_iter}')

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:48:47 2024

@author: joonc
"""

import numpy as np

# Given parameters
beta = np.array([
    [0.1901, 0.0660, 0.0028],
    [0.1084, 0.1550, 0.0453],
    [0.1356, 0.1540, 0.1165]
])

gamma = np.array([0.125, 0.083, 0.048])
N = np.array([6246073, 31530507, 13972160])  # [N1, N2, N3]

# Construct the Next Generation Matrix (K)
K = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        K[i, j] = beta[i, j] * (N[i] / N[j]) * (1.0 / gamma[j])

# Method 1: Direct eigenvalue computation
eigenvalues = np.linalg.eigvals(K)
R0_numpy = max(abs(eigenvalues))

# Method 2: Power Method to approximate the largest eigenvalue
def power_method(A, max_iter=1000, tol=1e-9):
    # Start with a random vector (or ones)
    x = np.ones(A.shape[0])
    x = x / np.linalg.norm(x)
    lambda_old = 0.0
    for iteration in range(1, max_iter+1):
        # Compute A*x
        Ax = A @ x
        # Estimate the eigenvalue from the ratio
        lambda_new = np.dot(x, Ax) / np.dot(x, x)
        # Normalize the vector
        x = Ax / np.linalg.norm(Ax)
        
        # Check for convergence
        if abs(lambda_new - lambda_old) < tol:
            return lambda_new, iteration
        lambda_old = lambda_new
    
    # If not converged within max_iter, return last estimate
    return lambda_new, max_iter

R0_power, num_iterations = power_method(K)

print("Next Generation Matrix (K):")
print(K)

print("\nEigenvalues of K (numpy):")
print(eigenvalues)
print(f"R0 from eigenvalues (numpy): {R0_numpy}")

print("\nPower Method Approximation:")
print(f"R0 from power method: {R0_power}")
print(f"Number of iterations (power method): {num_iterations}")

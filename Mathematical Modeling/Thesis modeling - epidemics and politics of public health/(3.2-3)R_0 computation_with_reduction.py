# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 20:32:10 2025

@author: joonc
"""

import numpy as np

# Given parameters
beta_original = np.array([
    [0.1901, 0.0660, 0.0028],
    [0.1084, 0.1550, 0.0453],
    [0.1356, 0.1540, 0.1165]
])

gamma = np.array([0.125, 0.083, 0.048])
N = np.array([6246073, 31530507, 13972160])  # [N1, N2, N3]

# Define the reduction level (percentage)
reduction_levels = [0, 0.1, 0.35, 0.7]  # 0%, 10%, 35%, 70% reductions

for reduction in reduction_levels:
    print(f"\nApplying {int(reduction * 100)}% reduction to beta:")
    # Apply the reduction uniformly to the entire beta matrix
    beta_reduced = beta_original * (1 - reduction)

    # Construct the Next Generation Matrix (K)
    K = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            K[i, j] = beta_reduced[i, j] * (N[i] / N[j]) * (1.0 / gamma[j])

    # Method 1: Direct eigenvalue computation
    eigenvalues = np.linalg.eigvals(K)
    R0_numpy = max(abs(eigenvalues))

    # Method 2: Power Method to approximate the largest eigenvalue
    def power_method(A, max_iter=1000, tol=1e-9):
        # Start with a random vector (or ones)
        x = np.ones(A.shape[0])
        x = x / np.linalg.norm(x)
        lambda_old = 0.0
        for iteration in range(1, max_iter + 1):
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

    # Output Results
    print("\nNext Generation Matrix (K):")
    print(K)

    print("\nEigenvalues of K (numpy):")
    print(eigenvalues)
    print(f"R0 from eigenvalues (numpy): {R0_numpy:.4f}")

    print("\nPower Method Approximation:")
    print(f"R0 from power method: {R0_power:.4f}")
    print(f"Number of iterations (power method): {num_iterations}")
    
    print("__________")

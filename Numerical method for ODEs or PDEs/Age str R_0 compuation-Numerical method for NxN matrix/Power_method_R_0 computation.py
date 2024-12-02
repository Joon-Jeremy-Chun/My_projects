# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 19:30:00 2024

@author: joonc
"""

import numpy as np
import pandas as pd
import time

# Step 1: Load the Next Generation Matrix (NGM)
NGM = pd.read_csv("NGM_800x800.csv").values

# Step 2: Define the Power Method with iteration count and timing
def power_method(matrix, max_iterations=1000, tolerance=1e-9):
    """
    Compute the largest eigenvalue (spectral radius) using the Power Method.
    Tracks iterations and time taken.
    :param matrix: The square matrix (numpy array)
    :param max_iterations: Maximum number of iterations
    :param tolerance: Convergence tolerance
    :return: Largest eigenvalue (spectral radius), corresponding eigenvector,
             number of iterations, and computation time
    """
    n = matrix.shape[0]
    b_k = np.random.rand(n)  # Start with a random vector
    b_k = b_k / np.linalg.norm(b_k)  # Normalize the vector

    start_time = time.time()  # Start timing
    for iteration in range(max_iterations):
        # Multiply the matrix by the vector
        b_k1 = np.dot(matrix, b_k)
        
        # Compute the eigenvalue (Rayleigh quotient)
        eigenvalue = np.dot(b_k1, b_k) / np.dot(b_k, b_k)
        
        # Normalize the resulting vector
        b_k1_norm = np.linalg.norm(b_k1)
        b_k1 = b_k1 / b_k1_norm
        
        # Check for convergence
        if np.linalg.norm(b_k1 - b_k) < tolerance:
            end_time = time.time()
            return eigenvalue, b_k1, iteration + 1, end_time - start_time
        
        b_k = b_k1

    end_time = time.time()
    return eigenvalue, b_k1, max_iterations, end_time - start_time

# Step 3: Apply the Power Method to the NGM
R_0, eigenvector, iterations, computation_time = power_method(NGM)

# Step 4: Output Results
print(f"Spectral Radius (R_0): {R_0}")
print(f"Number of Iterations: {iterations}")
print(f"Time Taken (seconds): {computation_time:.6f}")
print("\nDominant Eigenvector (First 10 values):")
print(eigenvector[:10])


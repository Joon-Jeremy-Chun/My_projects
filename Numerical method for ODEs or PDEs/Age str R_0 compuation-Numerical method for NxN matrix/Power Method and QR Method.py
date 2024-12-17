# Power method and QR method 
import numpy as np
import pandas as pd
import time

# Step 1: Load the Next Generation Matrix (NGM)
NGM = pd.read_csv("NGM_800x800.csv").values

# Step 2a: Define the Power Method with iteration count and timing
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

# Step 2b: Define the built-in computation for spectral radius
def compute_spectral_radius(matrix):
    """
    Compute the spectral radius of a matrix using numpy's eigenvalue computation.
    Tracks time taken for the computation.
    :param matrix: The square matrix (numpy array)
    :return: Spectral radius, computation time
    """
    start_time = time.time()  # Start timing
    eigenvalues = np.linalg.eigvals(matrix)  # Compute all eigenvalues
    spectral_radius = max(abs(eigenvalues))  # Compute the spectral radius
    end_time = time.time()  # End timing
    return spectral_radius, end_time - start_time

# Step 3: Compute Spectral Radius using both methods

# Method 1: Power Method
R_0_power, eigenvector, iterations, time_power = power_method(NGM)

# Method 2: Built-in Eigenvalue Computation
R_0_builtin, time_builtin = compute_spectral_radius(NGM)

# Step 4: Output Results
print("=== Results Using Power Method ===")
print(f"Spectral Radius (R_0): {R_0_power}")
print(f"Number of Iterations: {iterations}")
print(f"Time Taken (seconds): {time_power:.6f}")
print("\nDominant Eigenvector (First 10 values):")
print(eigenvector[:10])

print("\n=== Results Using Built-in Eigenvalue Computation ===")
print(f"Spectral Radius (R_0): {R_0_builtin}")
print(f"Time Taken (seconds): {time_builtin:.6f}")
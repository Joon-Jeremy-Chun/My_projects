# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:18:30 2024

@author: joonc
"""
import numpy as np

def identifiability_algorithm(m, N, sigma, mu, nu, epsilon, delta, p, q, r, gamma, f, p_bar, q_bar, r_bar, gamma_bar, f_bar):
    # Step 1: Input data
    n = 3  # Number of compartments; should match the length of parameter lists
    
    # Step 2: Canonical vectors
    canonical_vectors = np.eye(n * m)  # Create identity matrix for canonical vectors
    
    # Step 3: Construct system matrices A(p) and A(p_bar)
    def construct_A(p, q, r, gamma, f):
        A = np.zeros((n * m, n * m))
        # Fill in the matrix for the three compartments (based on the model structure)
        A[0, 0] = -sigma[0] - (f[0] + q[0])
        A[1, 1] = q[0] - gamma[0] + (1 - epsilon[0]) * f[0]
        A[2, 2] = r[0] - nu[0]
        
        A[3, 3] = p[1] - sigma[1] - f[1]
        A[4, 4] = gamma[0] + epsilon[0] * f[0] + q[1] - gamma[1] + (1 - epsilon[1]) * f[1]
        A[5, 5] = r[1] - nu[1]
        
        A[6, 6] = p[2] - f[2]
        A[7, 7] = gamma[1] + epsilon[1] * f[1] + q[2] - gamma[2] + f[2]
        A[8, 8] = r[2]
        
        # Add connections between compartments
        A[3, 0] = sigma[0]
        A[4, 1] = mu[0] + epsilon[0] * f[0]
        A[6, 3] = sigma[1]
        A[7, 4] = mu[1] + epsilon[1] * f[1]
        
        return A
    
    # Construct matrices A(p) and A(p_bar)
    A_p = construct_A(p, q, r, gamma, f)
    A_p_bar = construct_A(p_bar, q_bar, r_bar, gamma_bar, f_bar)
    
    # Step 4: Output of the system for different initial states
    def compute_output(A, b, x0):
        # Calculate x(1) = A * x(0) + b
        return np.dot(A, x0) + b
    
    # Iterate through the canonical vectors
    identified_params = []
    for i in range(3):  # Iterate only over the length of the parameter lists (three compartments)
        # Initial state x(0) = N * e_i
        x0 = N * canonical_vectors[i]
        b = np.zeros(n * m)  # Assuming b is zero (no external input)
        
        # Compute outputs
        x_p = compute_output(A_p, b, x0)
        x_p_bar = compute_output(A_p_bar, b, x0)
        
        # Compare outputs to identify parameters
        if np.allclose(x_p, x_p_bar):
            identified_params.append((p[i], q[i], r[i], gamma[i], f[i]))
    
    # Step 5: Save identified variable parameters
    print("Identified Parameters:", identified_params)
    
    return identified_params

# Example usage:
m = 3  # Number of age compartments
N = 100000  # Total population

# Example known rates
sigma = [0.01, 0.01]
mu = [0.01, 0.01]
nu = [0.01, 0.01]
epsilon = [0.01, 0.01]
delta = [0.01, 0.01]

# Initial guesses for parameters
p = [0.99, 0.95, 0.98]
q = [0.99, 0.95, 0.98]
r = [0.99, 0.95, 0.98]
gamma = [0.04, 0.06, 0.05]
f = [0.02, 0.008, 0.012]

# Alternative set of parameters for comparison
p_bar = [0.99, 0.95, 0.98]
q_bar = [0.99, 0.95, 0.98]
r_bar = [0.99, 0.95, 0.98]
gamma_bar = [0.04, 0.06, 0.05]
f_bar = [0.02, 0.008, 0.012]

# Run the algorithm
identified_params = identifiability_algorithm(m, N, sigma, mu, nu, epsilon, delta, p, q, r, gamma, f, p_bar, q_bar, r_bar, gamma_bar, f_bar)


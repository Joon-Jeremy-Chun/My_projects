# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 11:41:49 2024

@author: joonc
"""

import numpy as np

# Define the observed data from Table 2
observed_data = [
    [66701, 1680, 74, 15075, 986, 67, 4400, 964, 53],  # k = 1
    [66398, 1522, 140, 15428, 886, 126, 4505, 881, 100],  # k = 2
    [66090, 1385, 198, 15798, 796, 176, 4615, 797, 145],  # k = 3
    [65778, 1256, 248, 16141, 717, 220, 4723, 738, 179],  # k = 4
    [65456, 1144, 295, 16476, 642, 260, 4833, 670, 212],  # k = 5
    [65152, 1040, 336, 16791, 584, 294, 4947, 615, 241],  # k = 6
    [64836, 960, 370, 17105, 516, 326, 5066, 544, 277],  # k = 7
    [64521, 870, 400, 17408, 447, 347, 5185, 505, 290],  # k = 8
    [64210, 772, 429, 17705, 447, 368, 5310, 450, 309],  # k = 9
    [63892, 721, 449, 17891, 395, 386, 5439, 410, 327],  # k = 10
    [63580, 665, 468, 18264, 351, 401, 5543, 384, 342],  # k = 11
    [63269, 620, 484, 18529, 295, 415, 5679, 358, 354],  # k = 12
    [62960, 545, 498, 18800, 262, 428, 5580, 331, 365],  # k = 13
    [62645, 510, 509, 19043, 260, 430, 5930, 301, 372],  # k = 14
    [62351, 459, 519, 19298, 218, 440, 6067, 266, 382]   # k = 15
]


# Parameters known or given in the problem
m = 3  # Number of age compartments
N = 100000  # Total population

# Example known rates (need to be adjusted based on the data)
sigma = [0.01, 0.01, 0.02]
mu = [0.01, 0.01, 0.03]
vi = [0.01, 0.01, 0.04]
epsilon = [0.01, 0.01]
delta = [0.01, 0.01]

# Placeholder initial guesses for parameters (can be adjusted based on specific use cases)
p = [0.99, 0.95, 0.98]
q = [0.99, 0.95, 0.98]
r = [0.99, 0.95, 0.98]
gamma = [0.04, 0.06, 0.05]
f = [0.02, 0.008, 0.012]

# Alternative set of parameters for comparison (also placeholders)
p_bar = [0.99, 0.95, 0.98]
q_bar = [0.99, 0.95, 0.98]
r_bar = [0.99, 0.95, 0.98]
gamma_bar = [0.04, 0.06, 0.05]
f_bar = [0.02, 0.008, 0.012]

# Define the identifiability algorithm (use the previous code snippet)
def identifiability_algorithm(m, N, sigma, mu, vi, epsilon, delta, p, q, r, gamma, f, p_bar, q_bar, r_bar, gamma_bar, f_bar):
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
        A[2, 2] = r[0] - vi[0]
        
        A[3, 3] = p[1] - sigma[1] - f[1]
        A[4, 4] = gamma[0] + epsilon[0] * f[0] + q[1] - gamma[1] + (1 - epsilon[1]) * f[1]
        A[5, 5] = r[1] - vi[1]
        
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
    for i in range(n):  # Iterate only over the length of the parameter lists (three compartments)
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

# Apply the algorithm to the observed data
identified_params = identifiability_algorithm(m, N, sigma, mu, vi, epsilon, delta, p, q, r, gamma, f, p_bar, q_bar, r_bar, gamma_bar, f_bar)

# Output the identified parameters
print("Identified Parameters:", identified_params)

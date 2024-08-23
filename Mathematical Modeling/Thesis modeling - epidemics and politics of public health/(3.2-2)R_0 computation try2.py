# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 11:50:33 2024

@author: joonc
"""
import numpy as np

def construct_A(p, n):
    """Construct the system matrix A based on parameters p."""
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = p[i % len(p)]
    return A

def construct_output(A, b, x0, k):
    """Construct the output x_p(k) of the system."""
    x_p = np.dot(np.linalg.matrix_power(A, k), x0) + np.sum([np.dot(np.linalg.matrix_power(A, j), b) for j in range(k)], axis=0)
    return x_p

def identifiability_algorithm(m, N, sigma, mu, nu, epsilon, delta, observed_data):
    n = 3 * m  # Three compartments for each age group (S, I, R)
    canonical_vectors = np.eye(n)  # Identity matrix for canonical vectors
    
    # Initialize arrays to hold identified parameters
    identified_params = []
    
    for i in range(1, m+1):
        # Loop over each compartment
        for l in range(1, n+1):
            # For each l, construct the initial state x(0) = Ne_l
            x0 = N * canonical_vectors[l-1]
            b = np.zeros(n)  # Assume b is zero (no external input)
            
            for k in range(1, len(observed_data)):
                A_p = construct_A(observed_data[k], n)  # Construct A(p)
                x_p = construct_output(A_p, b, x0, k)  # Calculate x_p(k)
                
                # Debugging output
                #print(f"x_p(k={k}):", x_p)
                #print(f"Observed Data (k={k}):", observed_data[k])
                
                # Adjusting the tolerance for comparison
                if np.allclose(x_p, observed_data[k], rtol=1e-2, atol=1e-2):  # Check if x_p(k) matches observed data
                    identified_params.append(observed_data[k][l-1])
                    
    return identified_params

# Observed data: Table 2 from the paper
observed_data = np.array([
    [66701, 1680, 74, 10575, 986, 674, 2400, 964, 53],
    [66398, 1522, 140, 15428, 1286, 1505, 581, 100, 0],
    [66090, 1358, 198, 15798, 1796, 176, 4151, 797, 145],
    [65778, 1256, 248, 16147, 2177, 220, 4723, 733, 179],
    [65466, 1144, 295, 16746, 642, 680, 4833, 670, 212],
    [65152, 1040, 336, 16971, 594, 294, 4947, 615, 241],
    [64836, 960, 370, 17105, 516, 306, 5066, 544, 277],
    [64521, 870, 400, 17408, 447, 347, 5185, 505, 290],
    [64120, 772, 429, 17705, 487, 368, 5310, 450, 309],
    [63782, 721, 449, 17981, 395, 386, 5439, 410, 327],
    [63580, 665, 468, 18264, 381, 401, 5549, 384, 342],
    [63269, 620, 484, 18529, 292, 415, 5679, 358, 354],
    [62960, 565, 498, 18800, 282, 425, 5808, 331, 365],
    [62645, 510, 509, 19043, 260, 430, 5930, 301, 372],
    [62351, 459, 519, 19298, 218, 440, 6067, 266, 382]
])

# Example Parameters
m = 3  # Number of compartments (age groups)
N = 100000  # Total population

# Known rates (dummy values for the purpose of the example)
sigma = [0.01, 0.01, 0.01]
mu = [0.01, 0.01, 0.01]
nu = [0.01, 0.01, 0.01]
epsilon = [0.01, 0.01]
delta = [0.01, 0.01]

# Run identifiability algorithm
identified_params = identifiability_algorithm(m, N, sigma, mu, nu, epsilon, delta, observed_data)

print("Identified Parameters:", identified_params)


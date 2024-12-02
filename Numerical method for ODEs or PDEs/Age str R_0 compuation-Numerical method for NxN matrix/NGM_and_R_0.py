# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 19:00:00 2024

@author: joonc
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

# Step 1: Original 8x8 Contact Matrix
contact_matrix = np.array([
    [19.2, 4.8, 3.0, 3.7, 3.1, 3.1, 2.3, 1.4],
    [4.8, 42.4, 6.4, 5.4, 5.4, 5.3, 4.6, 1.7],
    [3.0, 6.4, 20.7, 9.2, 7.1, 6.3, 5.6, 0.9],
    [3.7, 5.4, 9.2, 16.9, 10.1, 7.4, 6.0, 1.1],
    [3.1, 5.4, 7.1, 10.1, 13.1, 10.4, 7.5, 2.1],
    [3.1, 5.3, 6.3, 7.4, 10.4, 10.3, 8.3, 3.2],
    [2.3, 4.6, 5.6, 6.0, 7.5, 8.3, 7.2, 3.2],
    [1.4, 1.7, 0.9, 1.1, 2.1, 3.2, 3.2, 7.2]
])

# Scale the contact matrix to a transmission matrix
transmission_matrix = contact_matrix * 0.01  # Scale by 0.01

# Step 2: Extend the Matrix to 800x800
upsampled_matrix = np.kron(transmission_matrix, np.ones((100, 100)))

# Normalize the extended matrix to ensure consistency
scaling_factor = 100  # Matrix was extended by 100x
normalized_matrix = upsampled_matrix / scaling_factor

# Step 3: Gaussian Smoothing (Optional)
sigma = 60  # Adjust smoothness as needed
smoothed_matrix = gaussian_filter(normalized_matrix, sigma=sigma, mode='constant', cval=0)

# Step 4: Population Data (Extended to 800 groups)
population_data = np.array([
    19426, 20235, 20737, 20754, 22959, 23027, 22233, 21650,
    21550, 21168, 21326, 21015, 17700, 17070, 14569, 9987,
    6530, 5856
])

# Group populations for 8 groups
grouped_population = [sum(population_data[i:i+2]) for i in range(0, len(population_data) - 2, 2)]
grouped_population.append(sum(population_data[-3:]))
grouped_population = np.array(grouped_population)

# Extend population sizes to 800 subgroups
extended_population = np.repeat(grouped_population / 100, 100)

# Step 5: Compute the F Matrix (F_ij = beta_ij * (N_i / N_j))
F = np.zeros((800, 800))
for i in range(800):
    for j in range(800):
        if extended_population[j] != 0:  # Avoid division by zero
            F[i, j] = smoothed_matrix[i, j] * (extended_population[i] / extended_population[j])
        else:
            F[i, j] = 0

# Step 6: Define Recovery Durations and Gamma Values
recovery_durations = [7, 8, 9, 10, 11, 12, 13, 14]  # Recovery durations in days
gamma_vector = np.repeat(1 / np.array(recovery_durations), 100)  # Repeat gamma values for 800 groups

# Step 7: Construct the V Matrix
V_matrix = np.diag(gamma_vector)  # Diagonal matrix with gamma values

# Compute the inverse of V
V_inverse = np.linalg.inv(V_matrix)

# Step 8: Compute the Next Generation Matrix (NGM)
NGM = np.dot(F, V_inverse)  # NGM = F * V^-1

# Step 9: Compute the Spectral Radius
eigenvalues = np.linalg.eigvals(NGM)
spectral_radius = max(abs(eigenvalues))

# Step 10: Save Results and Output
pd.DataFrame(NGM).to_csv("NGM_800x800.csv", index=False)

# Print results
print("Grouped Population Sizes (8 groups):")
print(grouped_population)

print("\nExtended Population Sizes (800 groups):")
print(extended_population[:10], "...")  # Show only first 10 entries

print("\nNext Generation Matrix (NGM):")
print(NGM[:5, :5])  # Show first 5x5 block

print(f"\nSpectral Radius (R_0): {spectral_radius}")

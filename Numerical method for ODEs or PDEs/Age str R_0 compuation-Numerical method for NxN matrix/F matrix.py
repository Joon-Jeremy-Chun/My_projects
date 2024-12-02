# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 17:13:29 2024

@author: joonc
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Step 1: Population Data Interpolation
population_data = np.array([
    19426, 20235, 20737, 20754, 22959, 23027, 22233, 21650,
    21550, 21168, 21326, 21015, 17700, 17070, 14569, 9987,
    6530, 5856
])

# Combine the groups into 8 larger groups
combined_population = [
    sum(population_data[:3]),    # Under 15 years
    sum(population_data[3:6]),   # 15 to 29 years
    sum(population_data[6:9]),   # 30 to 44 years
    sum(population_data[9:12]),  # 45 to 59 years
    sum(population_data[12:13]), # 60 to 64 years
    sum(population_data[13:14]), # 65 to 69 years
    sum(population_data[14:16]), # 70 to 79 years
    sum(population_data[16:])    # 80 years and over
]

# Normalize the combined population
total_population = sum(combined_population)
normalized_population = np.array(combined_population) / total_population

# Define original x points (8 groups) and new x points (800 entries)
x_original = np.linspace(0, 7, 8)
x_new = np.linspace(0, 7, 800)

# Polynomial interpolation for proportions
interp_func = interp1d(x_original, normalized_population, kind='cubic')
interpolated_ratios = interp_func(x_new)

# Scale back to match total population
scaled_interpolated_population = interpolated_ratios * total_population

# Save the interpolated population data
pd.DataFrame({'Population': scaled_interpolated_population}).to_csv("scaled_interpolated_population.csv", index=False)

# Step 2: Gaussian Smoothing for Contact Matrix
# Original 8x8 age-contact matrix
matrix = np.array([
    [19.2, 4.8, 3.0, 3.7, 3.1, 3.1, 2.3, 1.4],
    [4.8, 42.4, 6.4, 5.4, 5.4, 5.3, 4.6, 1.7],
    [3.0, 6.4, 20.7, 9.2, 7.1, 6.3, 5.6, 0.9],
    [3.7, 5.4, 9.2, 16.9, 10.1, 7.4, 6.0, 1.1],
    [3.1, 5.4, 7.1, 10.1, 13.1, 10.4, 7.5, 2.1],
    [3.1, 5.3, 6.3, 7.4, 10.4, 10.3, 8.3, 3.2],
    [2.3, 4.6, 5.6, 6.0, 7.5, 8.3, 7.2, 3.2],
    [1.4, 1.7, 0.9, 1.1, 2.1, 3.2, 3.2, 7.2]
])

# Upsample the 8x8 matrix to 800x800
upsampled_matrix = np.kron(matrix, np.ones((100, 100)))

# Apply Gaussian smoothing
sigma = 60
smoothed_matrix = gaussian_filter(upsampled_matrix, sigma=sigma, mode='constant', cval=0)

# Save the smoothed contact matrix
pd.DataFrame(smoothed_matrix).to_csv("gaussian_smoothed_800x800.csv", index=False)

# Step 3: Compute the F Matrix
# Load interpolated population and smoothed contact matrix
population_data = scaled_interpolated_population
contact_matrix = smoothed_matrix

# Initialize F matrix
F = np.zeros((800, 800))

# Calculate F_ij = beta_ij * (N_i / N_j)
for i in range(800):
    for j in range(800):
        if population_data[j] != 0:  # Avoid division by zero
            F[i, j] = contact_matrix[i, j] * (population_data[i] / population_data[j])
        else:
            F[i, j] = 0

# Save the F matrix
F_df = pd.DataFrame(F)
F_df.to_csv("F_matrix_800x800.csv", index=False)

# Step 4: Visualize Results
plt.figure(figsize=(10, 6))

# Visualize the smoothed contact matrix
plt.subplot(1, 2, 1)
plt.imshow(smoothed_matrix, cmap='viridis')
plt.colorbar()
plt.title("Smoothed Contact Matrix (800x800)")

# Visualize the F matrix
plt.subplot(1, 2, 2)
plt.imshow(F, cmap='viridis')
plt.colorbar()
plt.title("F Matrix (800x800)")

plt.tight_layout()
plt.show()

print("First 5x5 block of the F matrix:")
print(F[:5, :5])


# -*- coding: utf-8 -*-
"""
Created on Mon May  5 00:28:36 2024

@author: joonc
"""
import numpy as np

# Original matrix based on your image
original_matrix = np.array([
    [19.2, 4.8, 3.0, 7.1, 3.7, 3.1, 2.3, 1.4, 1.4],
    [4.8, 42.4, 6.4, 5.9, 7.5, 6.3, 1.8, 1.7, 1.7],
    [3.0, 6.4, 20.7, 9.2, 10.1, 6.8, 3.4, 0.9, 0.9],
    [3.7, 5.4, 9.2, 16.9, 13.1, 7.4, 2.6, 1.5, 1.5],
    [3.1, 5.0, 7.1, 10.1, 13.1, 10.4, 3.5, 1.8, 1.8],
    [2.3, 5.0, 6.3, 6.8, 7.4, 10.4, 7.5, 3.2, 3.2],
    [1.4, 1.8, 2.0, 2.6, 2.6, 3.5, 7.2, 7.2, 7.2],
    [1.4, 1.7, 0.9, 1.5, 2.1, 1.8, 3.2, 7.2, 7.2]
])

# Creating a 3x3 matrix
new_groups = [
    (0, 1),       # Rows for 0-19
    (2, 3, 4),    # Rows for 20-49
    (5, 6, 7)     # Rows for 50-80+
]

# Initialize an empty 3x3 matrix
reduced_matrix = np.zeros((3, 3))


# # Summing the corresponding rows and columns
# for i, rows in enumerate(new_groups):
#     for j, cols in enumerate(new_groups):
#         reduced_matrix[i, j] = original_matrix[rows, :][:, cols].sum()


# Summing the corresponding rows and columns
for i, rows in enumerate(new_groups):
    for j, cols in enumerate(new_groups):
        # Calculate the sum of the relevant elements
        total_sum = original_matrix[rows, :][:, cols].sum()
        # Count the number of elements combined
        num_elements = len(rows) * len(cols)
        # Calculate the average and assign it
        reduced_matrix[i, j] = total_sum / num_elements

# Print the resulting 3x3 matrix
print(reduced_matrix)

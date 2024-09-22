# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 12:43:59 2024

@author: joonc
"""
#Transimssion matrix modification. 
import numpy as np

# Assuming your matrix is named 'data_matrix' and is a numpy array
data_matrix = np.array([
    [19.2, 4.8, 3.0, 7.1, 3.7, 3.1, 2.3, 1.4, 1.4],
    [4.8, 42.4, 6.4, 5.4, 7.5, 6.3, 2.0, 1.7, 1.9],
    [3.0, 6.4, 20.7, 9.2, 10.1, 6.8, 3.4, 1.5, 2.1],
    [7.1, 5.4, 9.2, 16.9, 10.1, 7.4, 3.6, 3.5, 3.2],
    [3.7, 7.5, 10.1, 10.1, 13.1, 7.4, 2.6, 2.1, 2.1],
    [3.1, 6.3, 6.8, 7.4, 7.4, 10.4, 3.5, 1.8, 1.8],
    [2.3, 2.0, 3.4, 3.6, 2.6, 3.5, 7.5, 3.2, 3.2],
    [1.4, 1.7, 1.5, 3.5, 2.1, 1.8, 3.2, 7.2, 7.2],
    [1.4, 1.9, 2.1, 3.2, 2.1, 1.8, 3.2, 7.2, 7.2]
])

# Define the age groups as slices
age_groups = {
    '0-19': [0, 1],
    '20-49': [2, 3, 4],
    '50+': [5, 6, 7, 8]
}

# Initialize the new matrix
new_matrix_size = len(age_groups)
new_matrix = np.zeros((new_matrix_size, new_matrix_size))

# Sum and average the entries according to the new age groups
for i, (key_i, indices_i) in enumerate(age_groups.items()):
    for j, (key_j, indices_j) in enumerate(age_groups.items()):
        sub_matrix = data_matrix[np.ix_(indices_i, indices_j)]
        new_matrix[i, j] = np.mean(sub_matrix)

print("Newly aggregated and averaged matrix:")
print(new_matrix)


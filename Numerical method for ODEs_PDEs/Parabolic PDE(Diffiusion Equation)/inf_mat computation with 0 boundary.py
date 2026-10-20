# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:37:59 2024

@author: joonc
"""

import numpy as np

# Define a A, 3x3, matrix
A = np.array([[2, -1/2, 0],
              [-1/2, 2, -1/2],
              [0, -1/2, 2]])

# Compute the inverse of the matrix

A_inv = np.linalg.inv(A)

# Define a B, 3x3, matrix
B = np.array([[0, 1/2, 0],
              [1/2, 0, 1/2],
              [0, 1/2, 0]])

U_1 = A_inv @ B

print(U_1)

#%%
# Initialize the list to store U matrices
Us = [A_inv @ B]  # U_1 is the first element

# Iterative computation
for i in range(1, 10):  # Iterate from U_2 to U_10
    Us.append(A_inv @ B @ Us[-1])  # Append the new U matrix to the list


print(Us[0])  
print(Us[9])  

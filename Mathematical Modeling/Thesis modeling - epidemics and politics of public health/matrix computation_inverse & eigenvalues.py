# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 13:55:19 2024

@author: joonc
"""

import numpy as np

# Define the matrix
matrix = np.array([[4, 7], [2, 6]])


#%%
# Calculate the inverse of the matrix
try:
    inverse_matrix = np.linalg.inv(matrix)
    print("Inverse of the matrix:")
    print(inverse_matrix)
except np.linalg.LinAlgError:
    print("Matrix is singular and cannot be inverted.")

#%%

# Calculate the transpose of the matrix
transpose_matrix = np.transpose(matrix)
print("Transpose of the matrix:")
print(transpose_matrix)

#%%
# Calculate the eigenvalues of the matrix
eigenvalues, eigenvectors = np.linalg.eig(matrix)
print("\nEigenvalues of the matrix:")
print(eigenvalues)

print("\nEigenvectors of the matrix:")
print(eigenvectors)

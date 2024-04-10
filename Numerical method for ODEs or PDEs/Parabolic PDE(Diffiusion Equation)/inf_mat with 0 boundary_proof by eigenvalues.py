# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:54:16 2024

@author: joonc
"""

import numpy as np

# Define matrices A and B
A = np.array([[2, -1/2, 0],
              [-1/2, 2, -1/2],
              [0, -1/2, 2]])
B = np.array([[0, 1/2, 0],
              [1/2, 0, 1/2],
              [0, 1/2, 0]])

# Compute the inverse of A
A_inv = np.linalg.inv(A)

# Compute U_1
U_1 = A_inv @ B

# Compute eigenvalues and eigenvectors for U_1
eigenvalues, eigenvectors = np.linalg.eig(U_1)

# Diagonal matrix of eigenvalues
D = np.diag(eigenvalues)

# Matrix P of eigenvectors
P = eigenvectors

# Inverse of P
P_inv = np.linalg.inv(P)

# Verify diagonalization, A should be approximately equal to PDP^-1
A_reconstructed = P @ D @ P_inv

# Output the results
print(eigenvalues)
print(D)
print(P)
print(A_reconstructed)
print(U_1)

#insight: the eigenvalues or D matrix are all smaller than 1. which means the matrix approaches 0 as n goes to infinite
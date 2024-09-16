# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 12:01:59 2024

@author: joonc
"""

import numpy as np

def lu_decomposition(A):
    """
    Perform LU decomposition of matrix A.
    Returns matrices L and U such that A = LU.
    """
    n = A.shape[0]
    
    # Initialize L as the identity matrix and U as a copy of A
    L = np.eye(n)  # Lower triangular matrix
    U = A.copy()   # Upper triangular matrix (initially a copy of A)
    
    for i in range(n):
        # Perform Gaussian elimination
        for j in range(i+1, n):
            # Compute the multiplier for row elimination
            multiplier = U[j, i] / U[i, i]
            
            # Subtract the current row from the lower rows (elimination)
            U[j, i:] -= multiplier * U[i, i:]
            
            # Store the multiplier in L
            L[j, i] = multiplier
    
    return L, U

# Example of an n x n matrix
A = np.array([[2, -1, 3, 4],
              [4, 1, 2, -3],
              [3, 4, -1, 1],
              [1, 3, 2, 2]], dtype=float)

# Perform LU decomposition
L, U = lu_decomposition(A)

# Print the results
print("Matrix L (Lower triangular):")
print(L)
print("\nMatrix U (Upper triangular):")
print(U)

# Check
A_1 = L*U
print(A_1)

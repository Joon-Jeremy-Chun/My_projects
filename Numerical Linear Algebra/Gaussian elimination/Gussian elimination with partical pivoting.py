# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:07:03 2024

@author: joonc
"""

import numpy as np

def lu_decomposition_with_partial_pivoting(A):
    # Get the number of rows
    n = A.shape[0]
    
    # Initialize L, U, and P as identity matrices
    L = np.eye(n)
    U = A.copy()
    P = np.eye(n)

    for i in range(n):
        # Partial pivoting
        max_row = np.argmax(np.abs(U[i:, i])) + i
        if i != max_row:
            # Swap the rows in U
            U[[i, max_row]] = U[[max_row, i]]
            # Swap the rows in P
            P[[i, max_row]] = P[[max_row, i]]
            # Swap the rows in L (but only the columns before i)
            if i > 0:
                L[[i, max_row], :i] = L[[max_row, i], :i]
        
        # Eliminate entries below the pivot
        for j in range(i+1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:] -= L[j, i] * U[i, i:]

    return P, L, U

# Example usage:
A = np.array([[2, 1, -1, 3],
              [-2, 0, 0, 0],
              [4, 1, -2, 6],
              [-6, -1, 2, -3]], dtype=float)

P, L, U = lu_decomposition_with_partial_pivoting(A)

print("P matrix:")
print(P)
print("\nL matrix:")
print(L)
print("\nU matrix:")
print(U)

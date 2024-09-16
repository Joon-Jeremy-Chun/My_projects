# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:02:17 2024

@author: joonc
"""

import numpy as np

def gaussian_elimination(A, b):
    """
    Perform Gaussian elimination on the matrix A with the right-hand side b.
    The matrix A is assumed to be an n x n matrix, and b is the n x 1 vector.
    """
    n = len(b)
    
    # Augment the matrix A with the vector b
    Augmented = np.hstack((A, b.reshape(-1, 1)))

    # Forward elimination: transform to upper triangular form
    for i in range(n):
        # Pivot: make sure that Augmented[i][i] is non-zero (find maximum in the column)
        max_row = np.argmax(np.abs(Augmented[i:n, i])) + i
        if Augmented[max_row, i] == 0:
            raise ValueError("Matrix is singular and cannot be solved.")
        
        # Swap the row with the largest pivot element with the current row
        Augmented[[i, max_row]] = Augmented[[max_row, i]]

        # Eliminate entries below the pivot
        for j in range(i+1, n):
            ratio = Augmented[j, i] / Augmented[i, i]
            Augmented[j, i:] -= ratio * Augmented[i, i:]

    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Augmented[i, -1] - np.dot(Augmented[i, i+1:n], x[i+1:n])) / Augmented[i, i]
    
    return x

# Example of an n x n matrix
A = np.array([[2, -1, 3, 4],
              [4, 1, 2, -3],
              [3, 4, -1, 1],
              [1, 3, 2, 2]], dtype=float)

# Right-hand side vector
b = np.array([1, 2, 3, 4], dtype=float)

# Solve the system using Gaussian elimination
solution = gaussian_elimination(A, b)
print("Solution:", solution)

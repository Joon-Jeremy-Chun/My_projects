# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 13:19:44 2024

@author: joonc
"""

import sympy as sp

# Define the symbols
beta, gamma, mu = sp.symbols('beta gamma mu')

# Constructing the matrix V
V = sp.Matrix([
    [-gamma - mu, 0, 0],
    [mu, -gamma - mu, 0],
    [0, mu, -gamma - mu]
])

# Calculating the determinant of the matrix
determinant = V.det()

# Displaying the matrix and its determinant
print("Matrix V:")
sp.pprint(V)
print("\nDeterminant:")
sp.pprint(determinant)

# Calculating the inverse of the matrix if it is non-singular
if determinant != 0:
    inverse_matrix = V.inv()
    print("\nInverse Matrix:")
    sp.pprint(inverse_matrix)
else:
    print("\nThe matrix is singular and cannot be inverted.")
    inverse_matrix = None

# Define symbols with subscripts
beta_11, beta_12, beta_13, beta_21, beta_22, beta_23, beta_31, beta_32, beta_33 = sp.symbols('beta_11 beta_12 beta_13 beta_21 beta_22 beta_23 beta_31 beta_32 beta_33')
N_C, N_A, N_S = sp.symbols('N_C N_A N_S')

# Constructing the matrix F with subscripts
F = sp.Matrix([
    [beta_11,         beta_12*N_C/N_A, beta_13*N_C/N_A],
    [beta_21*N_C/N_A, beta_22,         beta_23*N_C/N_A],
    [beta_31*N_C/N_A, beta_32*N_C/N_A, beta_33]
])

# Displaying the matrix F
print("\nMatrix F:")
sp.pprint(F)

# Calculate the Next Generation Matrix (F * V^-1) if V is invertible
if inverse_matrix is not None:
    next_generation_matrix = F * inverse_matrix
    print("\nNext Generation Matrix (F * V^-1):")
    sp.pprint(next_generation_matrix)
    
    # Simplifying each entry of the Next Generation Matrix
    simplified_matrix = next_generation_matrix.applyfunc(sp.simplify)
    print("\nSimplified Next Generation Matrix:")
    sp.pprint(simplified_matrix)
    
    # Calculating the eigenvalues of the Simplified Next Generation Matrix
    eigenvalues = simplified_matrix.eigenvals()
    print("\nEigenvalues of the Simplified Next Generation Matrix:")
    sp.pprint(eigenvalues)
    
    # Printing each entry of the Simplified Next Generation Matrix one by one
    print("\nEntries of the Simplified Next Generation Matrix:")
    for i in range(simplified_matrix.rows):
        for j in range(simplified_matrix.cols):
            print(f"Entry ({i+1}, {j+1}):")
            sp.pprint(simplified_matrix[i, j])
else:
    print("\nCannot calculate the Next Generation Matrix because V is singular and cannot be inverted.")

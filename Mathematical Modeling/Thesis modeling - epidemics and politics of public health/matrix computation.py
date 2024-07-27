# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 13:19:44 2024

@author: joonc
"""

import sympy as sp

# Define the symbols
beta, gamma = sp.symbols('beta gamma')

# # Constructing the matrix
# matrix = sp.Matrix([
#     [0, beta, 0],
#     [0, -beta + gamma, 0],
#     [0, -gamma, 0]
# ])

# # Constructing the matrix
# matrix = sp.Matrix([
#     [0, 0, 0],
#     [beta, -beta + gamma, 0],
#     [0, -gamma, 0]
# ])

# Constructing the matrix
matrix = sp.Matrix([
    [ beta, 0],
    [ 0, gamma]
])


# Calculating the determinant of the matrix
determinant = matrix.det()

# Displaying the matrix and its determinant
sp.pprint(matrix)
print("\nDeterminant:")
sp.pprint(determinant)

# Calculating the inverse of the matrix if it is non-singular
if determinant != 0:
    inverse_matrix = matrix.inv()
    print("\nInverse Matrix:")
    sp.pprint(inverse_matrix)
else:
    print("\nThe matrix is singular and cannot be inverted.")


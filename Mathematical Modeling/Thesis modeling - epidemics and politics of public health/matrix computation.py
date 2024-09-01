# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 13:19:44 2024

@author: joonc
"""

import sympy as sp
#%%
# Define the symbols
# beta, gamma, mu = sp.symbols('beta gamma mu')

# # Constructing the matrix V
# V = sp.Matrix([
#     [-gamma - mu, 0, 0],
#     [mu, -gamma - mu, 0],
#     [0, mu, -gamma - mu]
# ])

# a,b,c,d,e = sp.symbols('a b c d e')

# V = sp.Matrix([
#     [a, 0, 0],
#     [d, b, 0],
#     [0, e, c]
# ])

a11,a21,a22,a32,a33 = sp.symbols('a11 a21 a22 a32 a33')

V = sp.Matrix([
    [a11,   0,   0],
    [a21, a22,   0],
    [  0, a32, a33]
])


# Calculating the determinant of the matrix
determinant = V.det()
inverse_matrix = V.inv()

# Displaying the matrix and its determinant
print("Matrix V:")
sp.pprint(V)
print("\nDeterminant:")
sp.pprint(determinant)
print("\nInverse Matrix:")
sp.pprint(inverse_matrix)

#%%
# Calculating the inverse of the matrix if it is non-singular
if determinant != 0:
    inverse_matrix = V.inv()
    print("\nInverse Matrix:")
    sp.pprint(inverse_matrix)
else:
    print("\nThe matrix is singular and cannot be inverted.")
    inverse_matrix = None
#%%
# # Define symbols with subscripts
# beta_11, beta_12, beta_13, beta_21, beta_22, beta_23, beta_31, beta_32, beta_33 = sp.symbols('beta_11 beta_12 beta_13 beta_21 beta_22 beta_23 beta_31 beta_32 beta_33')
# N_C, N_A, N_S = sp.symbols('N_C N_A N_C')

# # Constructing the matrix F with subscripts
# F = sp.Matrix([
#     [beta_11,         beta_12*N_C/N_A, beta_13*N_C/N_A],
#     [beta_21*N_C/N_A, beta_22,         beta_23*N_C/N_A],
#     [beta_31*N_C/N_A, beta_32*N_C/N_A, beta_33]
# ])

# Define symbols with subscripts
beta_11, beta_12, beta_13, beta_21, beta_22, beta_23, beta_31, beta_32, beta_33 = sp.symbols('beta_11 beta_12 beta_13 beta_21 beta_22 beta_23 beta_31 beta_32 beta_33')
N_1,N_2,N_3 = sp.symbols('N_1 N_2 N_3')
# Constructing the matrix F with subscripts
# F = sp.Matrix([
#     [beta_11*N_1, beta_12*N_1, beta_13*N_1],
#     [beta_21*N_2, beta_22*N_2, beta_23*N_2],
#     [beta_31*N_3, beta_32*N_3, beta_33*N_3]
# ])

F = sp.Matrix([
    [beta_11, beta_12, beta_13],
    [beta_21, beta_22, beta_23],
    [beta_31, beta_32, beta_33]
])

# Displaying the matrix F
print("\nMatrix F:")
sp.pprint(F)

# Calculate the product F * V^-1 if V is invertible
if inverse_matrix is not None:
    product_matrix = F * inverse_matrix
    print("\nProduct of F and V^-1:")
    sp.pprint(product_matrix)
    
    # Calculating the eigenvalues of F * V^-1
    eigenvalues = product_matrix.eigenvals()
    print("\nEigenvalues of F * V^-1:")
    sp.pprint(eigenvalues)
else:
    print("\nCannot calculate the product F * V^-1 because V is singular and cannot be inverted.")

#%%
print(eigenvalues)

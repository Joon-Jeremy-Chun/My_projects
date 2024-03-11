# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:44:54 2024

@author: joonc
"""
#Use Jacobi method by creating for loop 
import numpy as np

# Define the coefficients matrix (3x3)
A = np.array([[5, -2, 3],
              [-3, 9, 1],
              [2, -1, -7]])

# Define the constants vector
b = np.array([-1, 2, 3])
#%%
A_inv = np.linalg.inv(A)

print("Inverse of matrix A:")
print(A_inv)

# Compute the product of A_inv and b
result = np.dot(A_inv, b)

print("Result of A^-1 * b:")
print(result)
#%%
# Solve the system of equations with linalg.solve() function
x = np.linalg.solve(A, b)

print("Solution:")
print("x1 =", x[0])
print("x2 =", x[1])
print("x3 =", x[2])
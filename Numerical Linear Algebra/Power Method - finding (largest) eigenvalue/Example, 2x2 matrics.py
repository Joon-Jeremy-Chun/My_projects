# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:40:38 2024

@author: joonc
"""

import numpy as np

# Define a matrix A
A = np.array([[2, -12], [1, -5]])

# Initial guess for the eigenvector
v = np.array([1, 1])

# Number of iterations
iterations = 10

for i in range(iterations):
    # Multiply A by the current eigenvector approximation
    w = np.dot(A, v)
    print('iteratin :',i)
    print(f'Mu_{i} :', np.linalg.norm(w, np.inf))
    
    # Normalize the result using the infinity norm to get the next approximation
    v = w / np.linalg.norm(w, np.inf)
    print(f'V-{i} :', v)
    print(" ")

# Use the final eigenvector approximation to estimate the eigenvalue
eigenvalue_estimate = np.dot(v, np.dot(A, v)) / np.dot(v, v)

v, eigenvalue_estimate
print('eigenvetor : ', v)
print('eigentvalue : ', eigenvalue_estimate)

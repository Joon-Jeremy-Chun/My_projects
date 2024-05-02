# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:41:18 2024

@author: joonc
"""
import numpy as np

#Define a matrix A
#A = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

# Define a matrix A and A inverse
B = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
A = np.linalg.inv(B)

# Initial guess for the eigenvector
v = np.array([1, 1, 1])

# Number of iterations
iterations = 200

for i in range(iterations):
    # Multiply A by the current eigenvector approximation
    w = np.dot(A, v)
    print('iteratin :',i+1)
    print(f'Mu_{i} :', np.linalg.norm(w, np.inf))
    
    # Normalize the result using the infinity norm to get the next approximation
    v = w / np.linalg.norm(w, np.inf)
    print(f'V-{i} :', v)
    print(" ")

# Use the final eigenvector approximation to estimate the eigenvalue
eigenvalue_estimate = np.dot(v, np.dot(A, v)) / np.dot(v, v)
eig_es = np.linalg.norm(w, np.inf)
v, eigenvalue_estimate
print('----------')
print('eigenvetor : ', v)
print('mu_ : ', eig_es)
print('eigentvalue : ', eigenvalue_estimate)

print(w)

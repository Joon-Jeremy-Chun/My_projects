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
def jacobi_method(A, b, x0, tol=1e-6, max_iter=1000):
    n = len(b)
    x = np.copy(x0)
    x_new = np.zeros_like(x)
    
    for _ in range(max_iter):
        for i in range(n):
            sigma = np.dot(A[i, :], x) - A[i, i] * x[i]
            x_new[i] = (b[i] - sigma) / A[i, i]
        
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        
        x = np.copy(x_new)
    
    raise ValueError("Jacobi method did not converge within the maximum number of iterations.")

#%%
# Solve the system of equations with linalg.solve() function
x = np.linalg.solve(A, b)

print("Solution:")
print("x1 =", x[0])
print("x2 =", x[1])
print("x3 =", x[2])
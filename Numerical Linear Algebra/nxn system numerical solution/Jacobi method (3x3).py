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
def jacobi_method(A, b, x0, tol, max_iter):
    n = len(b)
    x = np.copy(x0)
    x_new = np.zeros_like(x)
    
    for iteration in range(max_iter):
        x_previous = np.copy(x_new)
        print(x_previous)
        for i in range(n):
            sigma = np.dot(A[i, :], x) - A[i, i] * x[i]
            x_new[i] = (b[i] - sigma) / A[i, i]
            
        print(f"Iteration {iteration + 1}: {x_new}")
        # Check convergence
        if np.linalg.norm(x_new - x_previous) < tol:
            print(f"Converged after {iteration + 1} iterations.")
            return x_new
        
        x = np.copy(x_new)
        
    print("Reached maximum number of iterations.")  
    return x_new
        #x = np.copy(x_new)

#%%
# Initial guess for the solution
x0 = np.zeros_like(b, dtype=float)

# Solve the system of equations using Jacobi method
solution = jacobi_method(A, b, x0, 10**(-6) , 100)

print("Solution using Jacobi method:")
print(solution)



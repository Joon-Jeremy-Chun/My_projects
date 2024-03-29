# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 22:14:23 2024

@author: joonc
"""
import sympy as sp
# Define the transition matrix P
P = sp.Matrix([
    [1/2, 1/3, 1/6],
    [0, 1/3, 2/3],
    [1/2, 0, 1/2]
])

# Find eigenvalues and eigenvectors
eigenvals = P.eigenvals()
eigenvects = P.eigenvects()

eigenvals, eigenvects
print(eigenvals)

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:31:30 2024

@author: joonc
"""

# Function to multiply two 2x2 matrices
def multiply_matrices(A, B):
    result = [[0, 0], [0, 0]]  # Initialize the result matrix with zeros

    # Perform matrix multiplication
    for i in range(2):
        for j in range(2):
            result[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j]

    return result

# Example matrices
S = [[0, 1],
     [1, 0]]

R = [[0, -1],
     [1, -1]]
#%%
# Multiply the matrices
result = multiply_matrices(S, R)

# Display the result
print("Result of A * B:")
for row in result:
    print(row)

#%%
RS = multiply_matrices(R, S)
RRS = multiply_matrices(R, RS)
#%%
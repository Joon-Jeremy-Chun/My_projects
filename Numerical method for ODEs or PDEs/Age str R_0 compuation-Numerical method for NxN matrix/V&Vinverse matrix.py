# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 17:24:18 2024

@author: joonc
"""

import numpy as np
import pandas as pd

# Step 1: Define gamma vector for 800 groups
# Example recovery durations (D, in days) for 8 broader age groups
recovery_durations = [7, 8, 9, 10, 11, 12, 13, 14]  # D_i for 8 groups

# Define the original age groups (8 groups) and new age groups (800 groups)
x_original = np.linspace(0, 7, 8)  # 8 groups
x_new = np.linspace(0, 7, 800)     # 800 groups

# Interpolate recovery durations for 800 groups
interpolated_durations = np.interp(x_new, x_original, recovery_durations)

# Calculate gamma for 800 groups (gamma = 1 / D)
gamma_vector = 1 / interpolated_durations

# Step 2: Create the V matrix
# V is a diagonal matrix with gamma_vector as diagonal entries
V_matrix = np.diag(gamma_vector)

# Step 3: Compute the inverse of V
V_inverse = np.linalg.inv(V_matrix)

# Step 4: Save results to CSV for future use
pd.DataFrame(V_matrix).to_csv("V_matrix_800x800.csv", index=False)
pd.DataFrame(V_inverse).to_csv("V_inverse_800x800.csv", index=False)

# Print a sample of V and its inverse
print("First 5 diagonal entries of V matrix:")
print(np.diag(V_matrix)[:5])

print("\nFirst 5 diagonal entries of V inverse matrix:")
print(np.diag(V_inverse)[:5])


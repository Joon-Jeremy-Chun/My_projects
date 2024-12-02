# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 18:49:56 2024

@author: joonc
"""

import numpy as np
import pandas as pd
import time

# Step 1: Load the Next Generation Matrix (NGM)
NGM = pd.read_csv("NGM_800x800.csv").values

# Step 2: Define the built-in computation for spectral radius
def compute_spectral_radius(matrix):
    """
    Compute the spectral radius of a matrix using numpy's eigenvalue computation.
    Tracks time taken for the computation.
    :param matrix: The square matrix (numpy array)
    :return: Spectral radius, computation time
    """
    start_time = time.time()  # Start timing
    eigenvalues = np.linalg.eigvals(matrix)  # Compute all eigenvalues
    spectral_radius = max(abs(eigenvalues))  # Compute the spectral radius
    end_time = time.time()  # End timing
    return spectral_radius, end_time - start_time
#%%
# Step 3: Compute the Spectral Radius
spectral_radius, computation_time = compute_spectral_radius(NGM)

# Step 4: Output Results
print(f"Spectral Radius (R_0): {spectral_radius}")
print(f"Time Taken (seconds): {computation_time:.6f}")


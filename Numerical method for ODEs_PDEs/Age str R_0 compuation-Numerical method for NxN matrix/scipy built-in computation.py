# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 18:55:49 2024

@author: joonc
"""

from scipy.sparse.linalg import eigs
import numpy as np
import pandas as pd
import time

# Step 1: Load the Next Generation Matrix (NGM)
NGM = pd.read_csv("NGM_800x800.csv").values

# Step 2: Compute the Dominant Eigenvalue using SciPy's eigs
start_time = time.time()  # Start timing
dominant_eigenvalue, dominant_eigenvector = eigs(NGM, k=1, which='LM')  # 'LM' = Largest Magnitude
end_time = time.time()  # End timing

# Step 3: Extract the Spectral Radius (Largest Eigenvalue Magnitude)
R_0 = np.abs(dominant_eigenvalue[0])  # Spectral radius is the absolute value of the dominant eigenvalue

# Step 4: Output Results
print(f"Spectral Radius (R_0): {R_0}")
print(f"Time Taken (seconds): {end_time - start_time:.6f}")
print("\nDominant Eigenvector (First 10 values):")
print(dominant_eigenvector[:10])

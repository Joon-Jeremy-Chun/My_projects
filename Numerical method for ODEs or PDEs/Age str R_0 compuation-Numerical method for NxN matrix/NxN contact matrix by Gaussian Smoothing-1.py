# Using Gaussian Smoothing

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Original 8x8 age-contact matrix
matrix = np.array([
    [19.2, 4.8, 3.0, 3.7, 3.1, 3.1, 2.3, 1.4],
    [4.8, 42.4, 6.4, 5.4, 5.4, 5.3, 4.6, 1.7],
    [3.0, 6.4, 20.7, 9.2, 7.1, 6.3, 5.6, 0.9],
    [3.7, 5.4, 9.2, 16.9, 10.1, 7.4, 6.0, 1.1],
    [3.1, 5.4, 7.1, 10.1, 13.1, 10.4, 7.5, 2.1],
    [3.1, 5.3, 6.3, 7.4, 10.4, 10.3, 8.3, 3.2],
    [2.3, 4.6, 5.6, 6.0, 7.5, 8.3, 7.2, 3.2],
    [1.4, 1.7, 0.9, 1.1, 2.1, 3.2, 3.2, 7.2]
])

# Step 1: Upsample the 8x8 matrix to 800x800
upsampled_matrix = np.kron(matrix, np.ones((100, 100)))

# Step 2: Apply Gaussian smoothing
sigma = 60  # Control the smoothness
smoothed_matrix = gaussian_filter(upsampled_matrix, sigma=sigma, mode='constant', cval=0)  # 'constant' padding with 0

# Step 3: Visualize the results
plt.figure(figsize=(10, 6))

# Original upsampled matrix
plt.subplot(1, 2, 1)
plt.imshow(upsampled_matrix, cmap='viridis')
plt.colorbar()
plt.title("Upsampled Matrix (800x800)")

# Smoothed matrix
plt.subplot(1, 2, 2)
plt.imshow(smoothed_matrix, cmap='viridis')
plt.colorbar()
plt.title(f"Gaussian Smoothed Matrix (σ={sigma})")

plt.tight_layout()
plt.show()

# Optional: Save the smoothed matrix to a CSV file
import pandas as pd
pd.DataFrame(smoothed_matrix).to_csv("gaussian_smoothed_800x800.csv", index=False)

# Print the first 5x5 block for reference
print("First 5x5 block of the smoothed matrix:")
print(smoothed_matrix[:5, :5])

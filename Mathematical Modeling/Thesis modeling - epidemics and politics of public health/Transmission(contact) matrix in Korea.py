# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 16:35:07 2024

@author: joonc
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images
color_bar_path = 'DataSets/bar.png'  # Replace with the correct file path
matrix_path = 'DataSets/matrix.png'  # Replace with the correct file path

color_bar = cv2.imread(color_bar_path)
matrix_img = cv2.imread(matrix_path)

# Convert both images from BGR (OpenCV format) to RGB (for Matplotlib visualization)
color_bar_rgb = cv2.cvtColor(color_bar, cv2.COLOR_BGR2RGB)
matrix_rgb = cv2.cvtColor(matrix_img, cv2.COLOR_BGR2RGB)

# Display the images to verify they are loaded correctly
plt.imshow(matrix_rgb)
plt.title('Loaded Transmission Matrix Image')
plt.show()
#%%
# Step 1: Extract colors from the color bar by sampling at regular intervals
num_samples = 1000  # Increase the number of samples from the color bar to improve precision

# Assuming the color bar values range from 0.1 to 0.8 in a linear scale
color_values = np.linspace(0.1, 0.8, num=num_samples)
bar_height = color_bar_rgb.shape[0]

# Sample colors along the height of the color bar
sampled_colors = []
for i in np.linspace(0, bar_height-1, num_samples, dtype=int):
    sampled_colors.append(color_bar_rgb[i, 0])  # Sample the first column (color gradient)

sampled_colors = np.array(sampled_colors)  # Convert to a numpy array
#%%
# Step 2: Define a function to map each pixel in the matrix to the closest color in the bar
def get_closest_value(color, color_map, value_map):
    """ Find the closest color in the color_map and return the corresponding value. """
    distances = np.sqrt(np.sum((color_map - color) ** 2, axis=1))
    closest_index = np.argmin(distances)
    return value_map[closest_index]

# Create a new matrix to store the mapped values based on the matrix image
mapped_values = np.zeros((matrix_rgb.shape[0], matrix_rgb.shape[1]))

# Compare each pixel in the matrix to the sampled colors from the color bar and assign a value
for i in range(matrix_rgb.shape[0]):
    for j in range(matrix_rgb.shape[1]):
        pixel_color = matrix_rgb[i, j]
        # Find the closest value for the current pixel color
        mapped_values[i, j] = get_closest_value(pixel_color, sampled_colors, color_values)
#%%
# Step 3: Create an 8x8 matrix by averaging blocks of pixels

# Check the matrix resolution
print(f"Matrix Image Resolution: {matrix_rgb.shape[0]}x{matrix_rgb.shape[1]}")

# Assuming the matrix is roughly square and we want to generate an 8x8 matrix
matrix_size = matrix_rgb.shape[0]  # Use the height (and assume width is similar)
num_groups = 8  # We want an 8x8 matrix

# Calculate the size of each block (adjusting based on the actual matrix resolution)
block_size = matrix_size // num_groups

# Create an empty 8x8 matrix to store the averages
final_matrix = np.zeros((num_groups, num_groups))

# Loop through each block to calculate the average value
for i in range(num_groups):
    for j in range(num_groups):
        # Define the block coordinates
        block = mapped_values[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
        # Calculate the average of the block and store it in the final matrix
        final_matrix[i, j] = np.mean(block)

# Step 4: Scale the matrix to the range of 0 to 700
min_value = final_matrix.min()
max_value = final_matrix.max()

# Scale factor to make the maximum value in the matrix equal to 700
scaling_factor = 700 / max_value
scaled_matrix = final_matrix * scaling_factor

# Step 5: Display the final scaled 8x8 matrix
print("8x8 Transmission Matrix (Scaled to 0–700):")
print(scaled_matrix)

# Step 6: Visualize the final 8x8 scaled matrix as a heatmap (optional)
plt.imshow(scaled_matrix, cmap='viridis', origin='lower')
plt.colorbar(label='Transmission Rate (0–700)')
plt.title('8x8 Transmission Matrix (Scaled to 0–700)')
plt.show()

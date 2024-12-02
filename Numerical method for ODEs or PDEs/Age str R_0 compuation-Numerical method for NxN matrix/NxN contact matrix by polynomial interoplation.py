# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 10:05:12 2024

@author: joonc
"""

#Using polynomial interpolation

import numpy as np
from scipy.interpolate import interp2d
import pandas as pd

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

# Define original x and y grid points (age groups)
x = np.linspace(0, 7, 8)  # Original 8 points
y = np.linspace(0, 7, 8)  # Original 8 points

# # Define new x and y grid points for the 80x80 matrix
x_new = np.linspace(0, 7, 80)
y_new = np.linspace(0, 7, 80)

# Define new x and y grid points for the 800x800 matrix
#x_new = np.linspace(0, 7, 800)
#y_new = np.linspace(0, 7, 800)

# Polynomial interpolation using interp2d
interp_func = interp2d(x, y, matrix, kind='cubic')  # 'cubic' for smooth interpolation
expanded_matrix = interp_func(x_new, y_new)

# Convert to DataFrame for better readability
expanded_matrix_df = pd.DataFrame(expanded_matrix)

# Display the resulting 80x80 matrix
import ace_tools as tools; tools.display_dataframe_to_user(name="80x80 Polynomial Interpolated Matrix", dataframe=expanded_matrix_df)

# Return the DataFrame for potential further usage
expanded_matrix_df

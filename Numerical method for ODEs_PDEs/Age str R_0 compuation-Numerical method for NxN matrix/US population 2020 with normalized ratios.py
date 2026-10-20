# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 15:14:54 2024

@author: joonc
"""

# Using Polynomial Interpolation with Normalized Ratios

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

# Original population data from the table (in thousands)
population_data = np.array([
    19426, 20235, 20737, 20754, 22959, 23027, 22233, 21650,
    21550, 21168, 21326, 21015, 17700, 17070, 14569, 9987,
    6530, 5856
])

# Step 1: Combine the groups into 8 larger groups
combined_population = [
    sum(population_data[:3]),    # Under 15 years
    sum(population_data[3:6]),   # 15 to 29 years
    sum(population_data[6:9]),   # 30 to 44 years
    sum(population_data[9:12]),  # 45 to 59 years
    sum(population_data[12:13]), # 60 to 64 years
    sum(population_data[13:14]), # 65 to 69 years
    sum(population_data[14:16]), # 70 to 79 years
    sum(population_data[16:])    # 80 years and over
]

# Step 2: Normalize the combined population
combined_population = np.array(combined_population)
total_population = combined_population.sum()
normalized_population = combined_population / total_population

# Define original x points (8 groups)
x_original = np.linspace(0, 7, 8)

# Define new x points for 80 entries
x_new = np.linspace(0, 7, 80)

# Step 3: Polynomial interpolation for proportions
interp_func = interp1d(x_original, normalized_population, kind='cubic')  # 'cubic' for smooth interpolation
interpolated_ratios = interp_func(x_new)

# Step 4: Scale back to match total population
scaled_interpolated_population = interpolated_ratios * total_population

# Step 5: Convert to DataFrame for readability
scaled_population_df = pd.DataFrame({'Population': scaled_interpolated_population})

# Save the resulting data to a CSV file
scaled_population_df.to_csv("scaled_interpolated_population.csv", index=False)

# Print the first few rows of the DataFrame
print(scaled_population_df.head())

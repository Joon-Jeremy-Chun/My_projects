# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 09:52:53 2024

@author: joonc
"""

import numpy as np
import pandas as pd

# Define the age-contact matrix as a 2D NumPy array
age_contact_matrix = np.array([
    [19.2, 4.8, 3.0, 3.7, 3.1, 3.1, 2.3, 1.4],
    [4.8, 42.4, 6.4, 5.4, 5.4, 5.3, 4.6, 1.7],
    [3.0, 6.4, 20.7, 9.2, 7.1, 6.3, 5.6, 0.9],
    [3.7, 5.4, 9.2, 16.9, 10.1, 7.4, 6.0, 1.1],
    [3.1, 5.4, 7.1, 10.1, 13.1, 10.4, 7.5, 2.1],
    [3.1, 5.3, 6.3, 7.4, 10.4, 10.3, 8.3, 3.2],
    [2.3, 4.6, 5.6, 6.0, 7.5, 8.3, 7.2, 3.2],
    [1.4, 1.7, 0.9, 1.1, 2.1, 3.2, 3.2, 7.2]
])

# Define age groups as row and column labels
age_groups = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']

# Create a Pandas DataFrame for better readability
matrix_df = pd.DataFrame(age_contact_matrix, index=age_groups[:-1], columns=age_groups[:-1])

# Display the matrix
import ace_tools as tools; tools.display_dataframe_to_user(name="Age-Contact Matrix", dataframe=matrix_df)

# Return the DataFrame for potential further usage
matrix_df

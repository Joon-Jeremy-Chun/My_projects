# -*- coding: utf-8 -*-
"""
Created on Sun May  1 20:42:44 2024

@author: joonc
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the age range, for example from 0 to 100 years
ages = np.linspace(0, 100, 500)  # 500 points from age 0 to 100

# Logistic function as given in your equation
def logistic_function(x):
    return 1 / (1 + np.exp(0.1 * (x - 80)))

# Calculate the logistic values for each age
logistic_values = logistic_function(ages)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(ages, logistic_values, label='Logistic Model')
plt.xlabel('Age')
plt.ylabel('Value')
plt.title('Logistic Function Over Age')
plt.ylim(0, 1)
plt.grid(True)
plt.show()


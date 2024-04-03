# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 17:35:41 2024

@author: joonc
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Define the parameter lambda
lmbda = 2

# Generate values for k (number of events)
k_values = np.arange(0, 10)  # Let's plot up to 16 events

# Calculate the corresponding probabilities using the Poisson PMF formula
poisson_probs = poisson.pmf(k_values, lmbda)

# Plot the Poisson distribution
plt.bar(k_values, poisson_probs, color='blue', alpha=0.7)
plt.title('Poisson Distribution with λ = 6')
plt.xlabel('Number of Events (k)')
plt.ylabel('Probability')
plt.xticks(k_values)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 11:24:57 2025

@author: joonc

Example of pointwise - uniform convergence
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the domain for x (we choose [0, 0.99] to avoid the singularity at x=1)
x = np.linspace(0, 0.99, 400)

# Create a DataFrame to store the results
df = pd.DataFrame({'x': x})

# Compute the partial sums for n = 1 to 30
for n in range(1, 31):
    # Calculate S_n(x) = sum_{k=0}^{n} x^k using a for loop (vectorized over x)
    S_n = sum(x**k for k in range(n+1))
    # Store the result in the DataFrame with a column named "n=<value>"
    df[f'n={n}'] = S_n

# Plot each partial sum with a rainbow colormap
plt.figure(figsize=(10, 6))
colors = plt.cm.rainbow(np.linspace(0, 1, 21))  # 21 distinct colors for n=10,11,...,30

for i, n in enumerate(range(1, 31)):
    plt.plot(df['x'], df[f'n={n}'], color=colors[i], label=f'n={n}')

# Plot the limit function for reference: S(x) = 1/(1-x)
limit_function = 1 / (1 - x)
plt.plot(x, limit_function, 'k--', linewidth=2, label='Limit: 1/(1-x)')

plt.title('Pointwise Convergence of the Geometric Series Partial Sums')
plt.xlabel('x')
plt.ylabel('$S_n(x)$')
plt.legend(title='Partial Sum Order', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


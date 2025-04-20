# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:46:19 2025

@author: joonc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the domain: x 
x = np.linspace(1, 2, 400)

# Create a DataFrame to store the results
df = pd.DataFrame({'x': x})

# Compute the partial sums for n =
for n in range(1, 10):
    S_n = sum(x/k for k in range(n+1))
    df[f'n={n}'] = S_n

# Plot each partial sum with a rainbow colormap
plt.figure(figsize=(10, 6))
colors = plt.cm.rainbow(np.linspace(0, 1, 21))  

for i, n in enumerate(range(1, 10)):
    plt.plot(df['x'], df[f'n={n}'], color=colors[i], label=f'n={n}')

# Plot the limit function: S(x) = 1/(1-x)
# limit_function = 1 / (1 - x)
# plt.plot(x, limit_function, 'k--', linewidth=2, label='Limit: 1/(1-x)')

plt.title('Uniform Convergence of the Geometric Series Partial Sums on [0,0.5]')
plt.xlabel('x')
plt.ylabel('$S_n(x)$')
plt.legend(title='Partial Sum Order', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

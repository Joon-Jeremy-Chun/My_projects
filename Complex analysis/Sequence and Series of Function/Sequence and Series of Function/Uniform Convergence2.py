# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:59:28 2025

@author: joonc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the domain for x
x = np.linspace(0, 10, 501)

# Create a DataFrame to store the computed values
df = pd.DataFrame({'x': x})

# Compute the functions f_n(x) = x/n for n from 1 to 30
for n in range(1, 31):
    df[f'f_{n}'] = x / n

# Plot each f_n(x) using a rainbow colormap
plt.figure(figsize=(10, 6))
colors = plt.cm.rainbow(np.linspace(0, 1, 30))  # 30 distinct colors

for i, n in enumerate(range(1, 31)):
    plt.plot(df['x'], df[f'f_{n}'], color=colors[i], label=f'n={n}')

# Plot the limit function: f(x) = 0
plt.plot(x, np.zeros_like(x), 'k--', linewidth=2, label='Limit: 0')

plt.title('Uniform Convergence of $f_n(x)=x/n$ on [0,10]')
plt.xlabel('x')
plt.ylabel('$f_n(x)=\\frac{x}{n}$')
plt.legend(title='Function Index', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

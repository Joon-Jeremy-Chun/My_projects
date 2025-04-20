# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:55:19 2025

@author: joonc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the domain for x (we use [0, 0.99] to avoid divergence at x = 1)
x = np.linspace(0, 1, 501)

# Create a DataFrame to store the computed partial sums
df = pd.DataFrame({'x': x})

# Compute the partial sums f_n(x)=sum_{k=1}^{n} x^k for n from 1 to 30
for n in range(1, 31):
    # f_n(x) is computed as the sum from k=1 to n of x^k
    
    f_n = x**n
   # f_n = sum(x**k for k in range(1, n+1))
    df[f'f_{n}'] = f_n

# Plot each partial sum using a rainbow colormap
plt.figure(figsize=(10, 6))
colors = plt.cm.rainbow(np.linspace(0, 1, 30))  # 30 distinct colors

for i, n in enumerate(range(1, 31)):
    plt.plot(df['x'], df[f'f_{n}'], color=colors[i], label=f'n={n}')

# Plot the limit function: f(x)=x/(1-x)
# limit_function = x / (1 - x)
# plt.plot(x, limit_function, 'k--', linewidth=2, label='Limit: x/(1-x)')

plt.title('Pointwise Convergence of the Geometric Series Partial Sums')
plt.xlabel('x')
plt.ylabel('$f_n(x) = \\sum_{k=1}^{n} x^k$')
plt.legend(title='Partial Sum Order', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

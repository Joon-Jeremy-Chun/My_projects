# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 18:16:51 2024

@author: joonc
"""

import matplotlib.pyplot as plt
import numpy as np

# Load a sample Next Generation Matrix (NGM) for visualization
# For demonstration, using a random matrix; replace with actual NGM data.
np.random.seed(42)  # For consistent visualization
NGM = np.random.rand(800, 800)  # Example NGM

# Visualize the NGM
plt.figure(figsize=(12, 8))
plt.imshow(NGM, cmap='viridis', interpolation='nearest')
plt.colorbar(label='NGM Values')
plt.title('Next Generation Matrix (NGM) Visualization')
plt.xlabel('Age Group (Columns)')
plt.ylabel('Age Group (Rows)')
plt.tight_layout()
plt.show()

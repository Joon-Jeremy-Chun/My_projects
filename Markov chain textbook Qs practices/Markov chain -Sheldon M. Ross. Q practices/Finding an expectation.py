# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 18:12:40 2024

@author: joonc
"""
#a Markov chain with three states and the transition probability matrix is provided.
#gives us the initial distribution where the probability of starting in state 0, p{x_0 = 0}, and state 1, p{x_0 = 1}, is both 1/4, and state 2, p{x_0 = 2} is 1/2
#Find E[x_3]
#%%

import numpy as np

# Transition matrix
P = np.array([[1/2, 1/3, 1/6],
              [0, 1/3, 2/3],
              [1/2, 0, 1/2]])

# Initial distribution
initial_distribution = np.array([1/4, 1/4, 1/2])

# Calculate the distribution at step 3
distribution_step_3 = initial_distribution @ P @ P @ P

# Expected value calculation at step 3
E_X3 = np.sum(np.arange(3) * distribution_step_3)
print(E_X3)
print(distribution_step_3)

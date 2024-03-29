# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 22:37:26 2024

@author: joonc
"""

import numpy as np

# Transition matrix
P = np.array([[-1, -1],
              [-5, -4]])

D = np.array([[2, 0],
              [0, 1]])

F = np.array([[4, -1],
              [-5, 1]])



p_3 =  P @ D @ F

print(p_3)

#%%
P = np.array([[-5, -4],
              [5, 5]])

D = np.array([[1, 0],
              [0, 2]])

F = np.array([[-1, -4/5],
              [1, 1]])


p_3 =  P @ D @ F

print(p_3)
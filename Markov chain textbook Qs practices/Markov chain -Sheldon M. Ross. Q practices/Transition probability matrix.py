# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 19:23:14 2024

@author: joonc
"""

import numpy as np

# Transition probability matrix
# P = np.array([[0.7, 0, 0.3, 0],
#             [0.5, 0, 0.5, 0],
#             [0, 0.4, 0, 0.6],
#             [0, 0.2, 0, 0.8]])

P = np.array([[0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.4, 0.5]])


v = np.array([1, 0, 0])

a = v @ P @ P @ P @ P@ P @ P @ P @ P @ P@ P @ P @P@ P @ P@ P @ P @ P@ P @ P @ P @ P @ P@ P @ P @P@ P @ P
print(a)
b = P @ P @ P @ P@ P
print(b)
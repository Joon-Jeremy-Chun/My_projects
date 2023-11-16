# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 08:33:45 2023

@author: joonc
"""

import pandas as pd
import matplotlib.pyplot as plt

#%%
# Given data x = time (s) y = speed (feet/second)
x = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84]
y = [124, 134, 148, 156, 147, 133, 121, 109, 99, 85, 78, 89, 104, 116, 123]

data = {
        'time' : x,
        'speed' : y 
        }

df = pd.DataFrame(data)

#%%

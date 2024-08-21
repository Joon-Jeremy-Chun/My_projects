# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:18:30 2024

@author: joonc
"""

import numpy as np
import pandas as pd

# Time steps (k = 1 to 15)
time_steps = np.arange(1, 16)

# Data for age compartment 1 (S1, I1, R1)
S1 = np.array([66701, 66398, 66090, 65778, 65466, 65152, 64836, 64521, 64210, 63892, 63580, 63269, 62960, 62645, 62351])
I1 = np.array([1680, 1522, 1385, 1256, 1144, 1040, 960, 870, 772, 721, 665, 620, 545, 510, 459])
R1 = np.array([74, 140, 198, 248, 295, 336, 370, 400, 429, 449, 468, 484, 498, 509, 519])

# Data for age compartment 2 (S2, I2, R2)
S2 = np.array([15075, 15442, 15798, 16141, 16476, 16791, 17105, 17408, 17705, 17981, 18264, 18529, 18800, 19043, 19298])
I2 = np.array([986, 886, 796, 717, 642, 584, 516, 474, 447, 395, 353, 292, 268, 260, 218])
R2 = np.array([126, 176, 220, 260, 294, 326, 347, 368, 386, 401, 415, 425, 430, 440, 440])

# Data for age compartment 3 (S3, I3, R3)
S3 = np.array([4400, 4505, 4615, 4723, 4835, 4947, 5066, 5185, 5310, 5439, 5543, 5679, 5808, 5930, 6067])
I3 = np.array([964, 881, 797, 738, 670, 615, 544, 505, 450, 410, 384, 358, 331, 301, 266])
R3 = np.array([53, 100, 145, 179, 212, 241, 277, 290, 309, 327, 342, 354, 365, 372, 382])

# Create a DataFrame for each age compartment
df1 = pd.DataFrame({
    'Time': time_steps,
    'S1': S1,
    'I1': I1,
    'R1': R1
})

df2 = pd.DataFrame({
    'Time': time_steps,
    'S2': S2,
    'I2': I2,
    'R2': R2
})

df3 = pd.DataFrame({
    'Time': time_steps,
    'S3': S3,
    'I3': I3,
    'R3': R3
})

# Display the DataFrames
print("Age Compartment 1:")
print(df1)
print("\nAge Compartment 2:")
print(df2)
print("\nAge Compartment 3:")
print(df3)

# Optionally, you can save these DataFrames to CSV files
df1.to_csv('age_compartment_1.csv', index=False)
df2.to_csv('age_compartment_2.csv', index=False)
df3.to_csv('age_compartment_3.csv', index=False)

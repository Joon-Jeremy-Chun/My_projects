# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:21:41 2023

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
# Create a line plot, Y for 'speed' and X for time

plt.plot(df['time'], df['speed'], label='speed of car', marker='o', markersize=2)
plt.xlabel('Time(s)')
plt.ylabel('Speed(feet/s)')
plt.legend()
plt.title('Plot : speed of car by time')
plt.grid(True)

#y-value temporally solve by hand
#plt.axhline(y=np.sqrt(3), color='red', linestyle='--', label='Horizontal Line at the root')


#plt.show()
plt.savefig('basic_plots.png')
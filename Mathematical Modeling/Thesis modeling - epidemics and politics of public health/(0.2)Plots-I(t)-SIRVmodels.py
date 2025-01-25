# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 17:25:38 2025

@author: joonc
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'DataSets/Korea_threeGroups_SIRV_parameter.csv'
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'])

# Filter the data for the required date range
start_date = '2020-01-25'
end_date = '2020-05-01'
data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

# Define age groups
age_groups = ['0-19', '20-49', '50-80+']

# Create a figure for each age group
for age_group in age_groups:
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data[f'I_{age_group}'], label=f'Infectious ({age_group})', color='blue')
    plt.title(f'Infectious Population (I(t)) for Age Group {age_group}')
    plt.xlabel('Date')
    plt.ylabel('Infectious Population')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Calculate the total infectious population
data['I_total'] = data[[f'I_{age_group}' for age_group in age_groups]].sum(axis=1)

# Plot the total infectious population
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['I_total'], label='Total Infectious Population', color='red')
plt.title('Total Infectious Population (I(t)) Across All Age Groups')
plt.xlabel('Date')
plt.ylabel('Total Infectious Population')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

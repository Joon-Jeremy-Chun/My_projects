# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 17:25:38 2025

@author: joonc
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'DataSets/Korea_threeGroups_SIRV_parameter.csv'  # Update this if needed
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'])

# Filter the data for the required date range
# start_date = '2020-01-25'
# end_date = '2020-05-01'
start_date = '2021-01-01'
end_date = '2021-12-30'



data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

# Define age groups
age_groups = ['0-19', '20-49', '50-80+']

# Create a figure for each age group with only one maximum point
for age_group in age_groups:
    plt.figure(figsize=(10, 6))

    # Plot infectious population
    plt.plot(data['Date'], data[f'I_{age_group}'], label=f'Infectious ({age_group})', color='blue')

    # Find the absolute maximum point
    max_value = data[f'I_{age_group}'].max()
    max_date = data[data[f'I_{age_group}'] == max_value]['Date'].values[0]

    # Plot the max point
    plt.scatter(max_date, max_value, color='red', zorder=3)
    plt.annotate(f'({pd.to_datetime(max_date).strftime("%Y-%m-%d")}, {int(max_value)})',
                 (max_date, max_value), textcoords="offset points", xytext=(-10,10),
                 ha='center', fontsize=10, color='red')

    # Formatting
    plt.title(f'Infectious Population (I(t)) for Age Group {age_group}')
    plt.xlabel('Date')
    plt.ylabel('Infectious Population')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Calculate the total infectious population
data['I_total'] = data[[f'I_{age_group}' for age_group in age_groups]].sum(axis=1)

# Plot the total infectious population with max point
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['I_total'], label='Total Infectious Population', color='red')

# Find the absolute maximum point for total
max_value_total = data['I_total'].max()
max_date_total = data[data['I_total'] == max_value_total]['Date'].values[0]

# Plot the max point
plt.scatter(max_date_total, max_value_total, color='black', zorder=3)
plt.annotate(f'({pd.to_datetime(max_date_total).strftime("%Y-%m-%d")}, {int(max_value_total)})',
             (max_date_total, max_value_total), textcoords="offset points", xytext=(-10,10),
             ha='center', fontsize=10, color='black')

# Formatting
plt.title('Total Infectious Population (I(t)) Across All Age Groups')
plt.xlabel('Date')
plt.ylabel('Total Infectious Population')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

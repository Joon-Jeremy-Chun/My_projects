# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 09:45:24 2024

@author: joonc
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# Create a folder 'Figures' if it doesn't exist
if not os.path.exists('Figures'):
    os.makedirs('Figures')

# Load the restructured data
file_path = 'DataSets/Korea_threeGroups_covid19_data.csv'  # Adjust the file path as necessary
data = pd.read_csv(file_path)

# Convert 'date' column to datetime format for better plotting
data['date'] = pd.to_datetime(data['date'])

# Set plot size
plt.figure(figsize=(10, 6))

# Time Series Plot: New Confirmed Cases over time for each age group
plt.subplot(2, 1, 1)
for age_group in data['new_age_group'].unique():
    age_group_data = data[data['new_age_group'] == age_group]
    plt.plot(age_group_data['date'], age_group_data['new_confirmed_cases'], label=age_group)
plt.title('Daily New Confirmed Cases by Age Group')
plt.xlabel('Date')
plt.ylabel('New Confirmed Cases')
plt.legend(title="Age Groups")
plt.grid(True)

# Cumulative Plot: Cumulative Confirmed Cases over time for each age group
plt.subplot(2, 1, 2)
for age_group in data['new_age_group'].unique():
    age_group_data = data[data['new_age_group'] == age_group]
    plt.plot(age_group_data['date'], age_group_data['confirmed_cases'].cumsum(), label=age_group)
plt.title('Cumulative Confirmed Cases by Age Group')
plt.xlabel('Date')
plt.ylabel('Cumulative Confirmed Cases')
plt.legend(title="Age Groups")
plt.grid(True)

# Save the time series and cumulative plot figure
plt.tight_layout()
plt.savefig('Figures/time_series_cumulative_cases.png')

# Show both plots
plt.show()

# Bar Plot: Total confirmed cases for each age group
total_cases = data.groupby('new_age_group')['confirmed_cases'].sum()

plt.figure(figsize=(8, 6))
total_cases.plot(kind='bar', color='skyblue')
plt.title('Total Confirmed Cases by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Total Confirmed Cases')
plt.xticks(rotation=45)
plt.grid(axis='y')

# Save the bar plot figure
plt.tight_layout()
plt.savefig('Figures/total_confirmed_cases_by_age_group.png')

# Show bar plot
plt.show()


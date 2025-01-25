# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 17:36:38 2025

@author: joonc
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the new cases data
file_path = 'DataSets\Korea_threeGroups_covid19_data.csv'
new_cases_data = pd.read_csv(file_path)

# Ensure proper datetime format and filter data between specified dates
new_cases_data['date'] = pd.to_datetime(new_cases_data['date'])
filtered_data = new_cases_data[(new_cases_data['date'] >= '2020-01-25') & (new_cases_data['date'] <= '2020-05-01')]

# Group data by date and age group to get the sum of new cases
grouped_data = filtered_data.groupby(['date', 'new_age_group']).agg({'new_confirmed_cases': 'sum'}).reset_index()

# Pivot data for easier plotting
pivoted_data = grouped_data.pivot(index='date', columns='new_age_group', values='new_confirmed_cases').fillna(0)
pivoted_data['Total'] = pivoted_data.sum(axis=1)

# Plot new cases for each age group and the total population
age_groups = ['0-19', '20-49', '50-80+']

# Individual plots for each age group
for age_group in age_groups:
    plt.figure(figsize=(10, 6))
    plt.plot(pivoted_data.index, pivoted_data[age_group], label=f"New Cases ({age_group})", color='blue')
    plt.title(f"New Confirmed Cases for Age Group {age_group}")
    plt.xlabel("Date")
    plt.ylabel("New Cases")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Total population plot
plt.figure(figsize=(10, 6))
plt.plot(pivoted_data.index, pivoted_data['Total'], label="New Cases (Total)", color='red')
plt.title("New Confirmed Cases for Total Population")
plt.xlabel("Date")
plt.ylabel("New Cases")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

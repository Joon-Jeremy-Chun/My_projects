# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 09:56:05 2024

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

# Calculate the rolling 14-day cumulative infections for each age group
data['rolling_14day_infections'] = data.groupby('new_age_group')['new_confirmed_cases'].rolling(window=14, min_periods=1).sum().reset_index(0, drop=True)

# Plot for each age group separately and save
for age_group in data['new_age_group'].unique():
    age_group_data = data[data['new_age_group'] == age_group]
    
    plt.figure(figsize=(10, 6))
    plt.plot(age_group_data['date'], age_group_data['rolling_14day_infections'], label=f'{age_group}')
    plt.title(f'14-Day Rolling Infected Cases for {age_group}')
    plt.xlabel('Date')
    plt.ylabel('Infected Cases (Rolling 14 Days)')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the figure in the 'Figures' folder
    plt.savefig(f'Figures/14day_rolling_infections_{age_group}.png')
    plt.show()

# Combined Plot for all age groups in one figure and save
plt.figure(figsize=(10, 6))
for age_group in data['new_age_group'].unique():
    age_group_data = data[data['new_age_group'] == age_group]
    plt.plot(age_group_data['date'], age_group_data['rolling_14day_infections'], label=age_group)
plt.title('14-Day Rolling Infected Cases (Combined Age Groups)')
plt.xlabel('Date')
plt.ylabel('Infected Cases (Rolling 14 Days)')
plt.legend(title="Age Groups")
plt.grid(True)
plt.tight_layout()

# Save the combined figure in the 'Figures' folder
plt.savefig('Figures/14day_rolling_infections_combined.png')
plt.show()


# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 09:56:05 2024

@author: joonc
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define start and end date
start_date = '2020-01-25'
end_date = '2020-05-01'

# Define rolling window size (change this value as needed)
rolling_window = 14  # Change to any number.

# Create a folder 'Figures' if it doesn't exist
if not os.path.exists('Figures'):
    os.makedirs('Figures')

# Load the restructured data
file_path = 'DataSets/Korea_threeGroups_covid19_data.csv'  # Adjust the file path as necessary
data = pd.read_csv(file_path)

# Convert 'date' column to datetime format for better plotting
data['date'] = pd.to_datetime(data['date'])

# Filter data within the required date range
data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

# Calculate the rolling X-day cumulative infections for each age group
data['rolling_infections'] = (
    data.groupby('new_age_group')['new_confirmed_cases']
    .rolling(window=rolling_window, min_periods=1).sum()
    .reset_index(0, drop=True)
)

# Plot for each age group separately and save
for age_group in data['new_age_group'].unique():
    age_group_data = data[data['new_age_group'] == age_group]

    plt.figure(figsize=(10, 6))
    plt.plot(age_group_data['date'], age_group_data['rolling_infections'], label=f'{age_group}', color='blue')

    # Find and annotate the absolute max point
    max_value = age_group_data['rolling_infections'].max()
    max_date = age_group_data[age_group_data['rolling_infections'] == max_value]['date'].values[0]
    plt.scatter(max_date, max_value, color='red', zorder=3)
    plt.annotate(f'({pd.to_datetime(max_date).strftime("%Y-%m-%d")}, {int(max_value)})',
                 (max_date, max_value), textcoords="offset points", xytext=(-10,10),
                 ha='center', fontsize=10, color='red')

    # Formatting
    plt.title(f'{rolling_window}-Day Rolling Infected Cases for {age_group}')
    plt.xlabel('Date')
    plt.ylabel(f'Infected Cases (Rolling {rolling_window} Days)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'Figures/{rolling_window}day_rolling_infections_{age_group}.png')
    plt.show()

# Calculate total rolling infections across all age groups
data_total = data.groupby('date')['rolling_infections'].sum().reset_index()

# Plot the total rolling infections with max point annotation
plt.figure(figsize=(10, 6))
plt.plot(data_total['date'], data_total['rolling_infections'], label='Total', color='red')

# Find and annotate the absolute max point
max_value_total = data_total['rolling_infections'].max()
max_date_total = data_total[data_total['rolling_infections'] == max_value_total]['date'].values[0]
plt.scatter(max_date_total, max_value_total, color='black', zorder=3)
plt.annotate(f'({pd.to_datetime(max_date_total).strftime("%Y-%m-%d")}, {int(max_value_total)})',
             (max_date_total, max_value_total), textcoords="offset points", xytext=(-10,10),
             ha='center', fontsize=10, color='black')

# Formatting
plt.title(f'{rolling_window}-Day Rolling Infected Cases (Total)')
plt.xlabel('Date')
plt.ylabel(f'Infected Cases (Rolling {rolling_window} Days)')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

# Save the figure
plt.savefig(f'Figures/{rolling_window}day_rolling_infections_total.png')
plt.show()


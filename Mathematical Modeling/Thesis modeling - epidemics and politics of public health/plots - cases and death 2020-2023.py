# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 01:23:39 2024

@author: joonc
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# Define output directory and age groups
output_dir = 'C:/Users/joonc/Documents'
age_groups = ['80 이상', '70-79', '60-69', '50-59', '40-49', '30-39', '20-29', '10-19', '0-9']

# Colors for each age group in rainbow order starting with red for '80 이상'
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'cyan', 'magenta']

# Set font properties to support Hangul characters
font_path = 'C:/Windows/Fonts/malgun.ttf'  # Change this to the path of the font on your system
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

# Function to plot new cases or new deaths for each age group
def plot_data(data_type):
    plt.figure(figsize=(15, 7))
    
    # Determine the full range of dates
    all_dates = pd.date_range(start='2020-01-20', end='2023-07-30')
    
    for age_group, color in zip(age_groups, colors):
        file_path = os.path.join(output_dir, f'covid19_data_20200120_to_20230731_{age_group}_processed.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.drop_duplicates(subset='date')  # Remove duplicate dates
            df = df.set_index('date').reindex(all_dates, fill_value=0).reset_index()
            df.rename(columns={'index': 'date'}, inplace=True)
            plt.bar(df['date'], df[data_type], color=color, label=age_group, alpha=0.6)
        else:
            print(f"File not found: {file_path}")
    
    plt.xlabel('Date')
    plt.ylabel(f'New {data_type.capitalize()}')
    plt.title(f'New {data_type.capitalize()} by Age Group')
    
    # Set x-axis labels to quarterly
    plt.xticks(pd.date_range(start='2020-01-20', end='2023-07-30', freq='Q'), rotation=45)
    plt.legend(title='Age Group')
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(output_dir, f'new_{data_type}_by_age_group.png')
    plt.savefig(plot_file)
    plt.show()
    print(f"Plot saved to {plot_file}")

# Plot new cases
plot_data('new_cases')

# Plot new deaths
plot_data('new_deaths')



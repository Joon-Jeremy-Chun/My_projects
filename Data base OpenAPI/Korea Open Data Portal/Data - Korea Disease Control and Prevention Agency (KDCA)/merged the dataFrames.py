# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 22:08:10 2024

@author: joonc
"""
#merged the dataFrames
import pandas as pd
import os

# Define base directory
base_dir = 'C:/Users/joonc/My_github/My_projects/Data base OpenAPI/Korea Open Data Portal/Data - Korea Disease Control and Prevention Agency (KDCA)'

# Define age groups and file patterns
age_groups = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+', '남성', '여성']
file_patterns = [
    'covid19_data_20200120_to_20210119_{}.csv',
    'covid19_data_20210120_to_20220119_{}.csv',
    'covid19_data_20220120_to_20230119_{}.csv',
    'covid19_data_20230120_to_20230730_{}.csv'
]

# Function to process and merge files for a specific age group
def process_and_merge_files_for_age_group(age_group):
    df_list = []
    for pattern in file_patterns:
        file_path = os.path.join(base_dir, pattern.format(age_group))
        if os.path.exists(file_path):
            print(f"Reading file: {file_path}")
            df_list.append(pd.read_csv(file_path))
        else:
            print(f"File not found: {file_path}")
    
    if df_list:
        merged_df = pd.concat(df_list, ignore_index=True)
        
        # Calculate new cases and new deaths
        merged_df['new_cases'] = merged_df['confirmed_cases'].diff().fillna(0).astype(int)
        merged_df['new_deaths'] = merged_df['deaths'].diff().fillna(0).astype(int)
        
        # Drop the redundant date column
        merged_df = merged_df.drop(columns=['date.1'])
        
        # Save the processed DataFrame to a new CSV file
        output_file = os.path.join(base_dir, f'covid19_data_20200120_to_20230731_{age_group}_processed.csv')
        merged_df.to_csv(output_file, index=False)
        print(f"Data for age group {age_group} processed and saved to {output_file}")
    else:
        print(f"No files found for age group {age_group}")

# Process and merge files for all age groups
for age_group in age_groups:
    process_and_merge_files_for_age_group(age_group)

# Merge all processed age group data into a single file
all_data = []
for age_group in age_groups:
    processed_file_path = os.path.join(base_dir, f'covid19_data_20200120_to_20230731_{age_group}_processed.csv')
    if os.path.exists(processed_file_path):
        print(f"Appending data from: {processed_file_path}")
        all_data.append(pd.read_csv(processed_file_path))
    else:
        print(f"File not found: {processed_file_path}")

if all_data:
    final_merged_df = pd.concat(all_data, ignore_index=True)
    final_output_file = os.path.join(base_dir, 'covid19_data_20200120_to_20230731_all_ages_processed.csv')
    final_merged_df.to_csv(final_output_file, index=False)
    print(f"All processed age group data merged and saved to {final_output_file}")
else:
    print("No data found for merging all processed age groups")




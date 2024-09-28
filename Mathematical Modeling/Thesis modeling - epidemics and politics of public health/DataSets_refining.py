import pandas as pd

# Define the age groups
age_groups = [
    '0-9', '10-19', '20-29', '30-39', '40-49', '50-59', 
    '60-69', '70-79', '80 이상'
]

# Define the time periods
time_periods = [
    '20200120_to_20210119', 
    '20210120_to_20220119', 
    '20220120_to_20230119', 
    '20230120_to_20230730'
]

# Initialize an empty list to hold combined data for all age groups
combined_dataframes = []

# Loop through each age group and combine the files for different time periods
for age_group in age_groups:
    # List of file paths for this specific age group
    file_paths = [f'DataSets/covid19_data_{period}_{age_group}.csv' for period in time_periods]
    
    # Load the datasets into pandas DataFrames
    dataframes = [pd.read_csv(file) for file in file_paths]
    
    # Concatenate all DataFrames for this age group into one
    combined_age_group_data = pd.concat(dataframes, ignore_index=True)
    
    # Append the combined data for this age group to the list
    combined_dataframes.append(combined_age_group_data)

# Concatenate all age groups into one final DataFrame, keeping age group labels intact
final_combined_data = pd.concat(combined_dataframes, ignore_index=True)

# Sort the data by date to ensure correct calculation of new confirmed cases
final_combined_data = final_combined_data.sort_values(by=['age_group', 'date'])

# Calculate the daily new confirmed cases by subtracting the previous day's cumulative total from the current day's total
final_combined_data['new_confirmed_cases'] = final_combined_data.groupby('age_group')['confirmed_cases'].diff().fillna(0).astype(int)

# Ensure no negative values in the new cases column (in case of data inconsistencies)
final_combined_data['new_confirmed_cases'] = final_combined_data['new_confirmed_cases'].clip(lower=0)

# Save the combined data with the new cases column into a new CSV file
final_combined_data.to_csv('DataSets/final_combined_covid19_data_with_new_cases.csv', index=False)

# Display the first few rows of the final combined dataset with new cases
print(final_combined_data[['age_group', 'date', 'confirmed_cases', 'new_confirmed_cases']].head())


#%%
#DataSet test
# Load the dataset
file_path = 'DataSets/final_combined_covid19_data_with_new_cases.csv'  # Adjust this to your file path
data = pd.read_csv(file_path)

# Filter for rows where 'new_confirmed_cases' has negative values
negative_cases = data[data['new_confirmed_cases'] < 0]

# If there are negative values, print them
if not negative_cases.empty:
    print("Negative values found in 'new_confirmed_cases':")
    print(negative_cases[['age_group', 'date', 'new_confirmed_cases']])
else:
    print("No negative values found in 'new_confirmed_cases'.")
    
#%%
#In three age groups
# Load the dataset
file_path = 'DataSets/final_combined_covid19_data_with_new_cases.csv'
data = pd.read_csv(file_path)

# Define new age group ranges and the corresponding mapping
def map_age_group(age_group):
    if age_group in ['0-9', '10-19']:
        return '0-19'
    elif age_group in ['20-29', '30-39', '40-49']:
        return '20-49'
    elif age_group in ['50-59', '60-69', '70-79', '80 이상']:
        return '50-80+'
    else:
        return None

# Apply the mapping function to create a new column with the combined age groups
data['new_age_group'] = data['age_group'].apply(map_age_group)

# Group the data by the new age groups and aggregate confirmed cases and new confirmed cases
grouped_data = data.groupby(['new_age_group', 'date'], as_index=False).agg({
    'confirmed_cases': 'sum',
    'new_confirmed_cases': 'sum'
})

# Save the restructured data into a new CSV file
grouped_data.to_csv('DataSets/Korea_threeGroups_covid19_data.csv', index=False)

# Display the first few rows of the restructured data
print(grouped_data.head())



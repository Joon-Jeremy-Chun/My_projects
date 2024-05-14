# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 21:53:47 2024

@author: joonc
"""

import requests
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime, timedelta

# API request URL
base_url = 'https://apis.data.go.kr/1352000/ODMS_COVID_05/callCovid05Api?serviceKey=VH1eV880GIQeDHmoQ61K1U%2BaOL4tjTbBKHVFwEL%2FEoPPYM6s1Ua7aYaJiRMABBzIlBPwFE2aR8Rs7CTJYcNyqQ%3D%3D'

# Request parameters
params = {
    
    'numOfRows': '500',   # Number of results per page
    'pageNo': '1',         # Page number
    'apiType': 'xml'       # Response format (xml)
}

# Date range settings
start_date = datetime.strptime('2022-01-01', '%Y-%m-%d')
end_date = datetime.strptime('2022-01-07', '%Y-%m-%d')

all_items = []

# Column name translation from Korean to English
column_name_mapping = {
    'confCase': 'confirmed_cases',
    'confCaseRate': 'confirmed_case_rate',
    'criticalRate': 'critical_rate',
    'death': 'deaths',
    'deathRate': 'death_rate',
    'gubun': 'age_group',
    'createDt': 'date'
}

# Collect data for each date in the date range
current_date = start_date
while current_date <= end_date:
    params['create_dt'] = current_date.strftime('%Y-%m-%d')
    response = requests.get(base_url, params=params)

    # 응답 확인
    if response.status_code == 200:
        xml_data = response.content  # XML response
        # Parse XML data
        root = ET.fromstring(xml_data)
        
        # Extract items
        for item in root.findall('.//item'):
            parsed_item = {child.tag: child.text for child in item}
            parsed_item['date'] = current_date.strftime('%Y-%m-%d')  # Add date information
            all_items.append(parsed_item)
        
        print(f"Data collected for {current_date.strftime('%Y-%m-%d')}")
    else:
        print(f"Error {response.status_code}: {response.text}")
    
    # Move to the next date
    current_date += timedelta(days=1)

# Convert collected data to a DataFrame
df = pd.DataFrame(all_items)

# Rename columns to English
df.rename(columns=column_name_mapping, inplace=True)

# Group data by 'age_group'
df_by_age = df.groupby('age_group')

# Save data for each age group to separate CSV files
for age_group, data in df_by_age:
    filename = f"covid19_data_20220101_to_20220107_{age_group}.csv"
    data.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"Data for age group {age_group} has been saved to {filename}")

# Save all data to a single CSV file (Ensure UTF-8 encoding)
df.to_csv("covid19_data_20220101_to_20220107.csv", index=False, encoding='utf-8-sig')

# Completion message
print("All data has been saved to covid19_data_20220101_to_20220107.csv")
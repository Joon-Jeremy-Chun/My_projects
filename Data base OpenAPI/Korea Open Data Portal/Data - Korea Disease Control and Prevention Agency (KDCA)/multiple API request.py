# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 23:08:05 2024

@author: joonc
"""

import requests
from datetime import datetime, timedelta

def fetch_data(date):
    url = "http://apis.data.go.kr/1352000/ODMS_COVID_04/callCovid04Api"
    params = {
        'serviceKey': 'YOUR_SERVICE_KEY',  # Replace with your URL-encoded service key
        'numOfRows': 10,  
        'pageNo': 1,
        'apiType': 'json', 
        'std_day': date.strftime('%Y-%m-%d'),
        'gubun': '경기'  # Replace with the desired region
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        try:
            return response.json()
        except ValueError:
            print("Error decoding JSON for date: ", date)
            return None
    else:
        print("Failed to retrieve data for date: ", date, response.status_code)
        return None

# Set the date range
start_date = datetime.strptime('2020-04-01', '%Y-%m-%d')
end_date = datetime.strptime('2021-04-01', '%Y-%m-%d')
current_date = start_date

# Dictionary to hold all the data
all_data = {}

# Loop through each day in the range
while current_date <= end_date:
    data = fetch_data(current_date)
    if data:
        all_data[current_date.strftime('%Y-%m-%d')] = data
    current_date += timedelta(days=1)

# Print or process all collected data
for date, data in all_data.items():
    print(date, data)

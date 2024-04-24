# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 23:06:07 2024

@author: joonc
"""

import requests

# Set the base URL of the API
url = "http://apis.data.go.kr/1352000/ODMS_COVID_04/callCovid04Api"

# Your service key needs to be URL-encoded if it contains special characters
service_key = 'YOUR_SERVICE_KEY'  # Use your actual service key here, properly URL-encoded

# Set parameters for the API request
params = {
    'serviceKey': service_key,
    'numOfRows': 500,   # You can adjust this as needed
    'pageNo': 1,        # Page number
    'apiType': 'json',  # Can be 'xml' or 'json' depending on what you prefer
    'std_day': '2021-12-15',  # Example date, change it to the date you are interested in
    'gubun': '경기'     # Example region, change to the city or region you want to query
}

# Make the HTTP GET request
response = requests.get(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    try:
        data = response.json()
        print(data)  # Print the data retrieved
    except ValueError:  # Includes simplejson.decoder.JSONDecodeError
        print("Failed to decode JSON from response: ", response.text)
else:
    print("Failed to retrieve data: ", response.status_code, response.text)


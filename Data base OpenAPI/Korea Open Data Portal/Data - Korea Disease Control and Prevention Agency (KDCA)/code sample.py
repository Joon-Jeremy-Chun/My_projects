# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 21:53:47 2024

@author: joonc
"""

import requests

url = 'http://apis.data.go.kr/1352000/ODMS_COVID_04/callCovid04Api'
params ={'serviceKey' : '서비스키', 'pageNo' : '1', 'numOfRows' : '500', 'apiType' : 'xml', 'std_day' : '2021-12-15', 'gubun' : '경기' }

response = requests.get(url, params=params)
print(response.content)
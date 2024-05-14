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
    
    'numOfRows': '1000',   # 한 페이지 결과 수
    'pageNo': '1',         # 페이지 번호
    'apiType': 'xml'       # 결과 형식 (xml)
}

# Date range settings
start_date = datetime.strptime('2022-01-01', '%Y-%m-%d')
end_date = datetime.strptime('2022-01-07', '%Y-%m-%d')

all_items = []

# 날짜 범위 내의 각 날짜에 대해 데이터 수집
current_date = start_date
while current_date <= end_date:
    params['create_dt'] = current_date.strftime('%Y-%m-%d')
    response = requests.get(base_url, params=params)

    # 응답 확인
    if response.status_code == 200:
        xml_data = response.content  # XML 형식의 응답일 경우
        # XML 데이터 파싱
        root = ET.fromstring(xml_data)
        
        # 아이템 추출
        for item in root.findall('.//item'):
            parsed_item = {child.tag: child.text for child in item}
            parsed_item['date'] = current_date.strftime('%Y-%m-%d')  # 추가: 날짜 정보
            all_items.append(parsed_item)
        
        print(f"Data collected for {current_date.strftime('%Y-%m-%d')}")
    else:
        print(f"Error {response.status_code}: {response.text}")
    
    # 다음 날짜로 이동
    current_date += timedelta(days=1)

# 모든 날짜의 데이터를 DataFrame으로 변환
df = pd.DataFrame(all_items)

# 'gubun' (age group)별로 데이터를 그룹화
df_by_age = df.groupby('gubun')

# 각 그룹에 대해 데이터를 확인하고 CSV 파일로 저장
for age_group, data in df_by_age:
    filename = f"covid19_data_20220101_to_20220107_{age_group}.csv"
    data.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"Data for age group {age_group} has been saved to {filename}")

# 전체 데이터를 하나의 CSV 파일로 저장 (Ensure UTF-8 encoding)
df.to_csv("covid19_data_20220101_to_20220107.csv", index=False, encoding='utf-8-sig')

# CSV 파일 저장 완료 메시지
print("All data has been saved to covid19_data_20220101_to_20220107.csv")


# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 21:53:47 2024

@author: joonc
"""

import requests
import xml.etree.ElementTree as ET
import pandas as pd


# API 요청 URL
url = ''

# 요청 파라미터
params = {
    'numOfRows': '10',      # 한 페이지 결과 수
    'pageNo': '1',          # 페이지 번호
    'apiType': 'xml',       # 결과 형식 (xml)
    'create_dt': '2022-01-01'  # 데이터 기준일
}

# API 요청
response = requests.get(url, params=params)

# 응답 확인
if response.status_code == 200:
    xml_data = response.content  # XML 형식의 응답일 경우
    # XML 데이터 파싱
    root = ET.fromstring(xml_data)
    
    # 아이템 추출
    items = []
    for item in root.findall('.//item'):
        parsed_item = {child.tag: child.text for child in item}
        items.append(parsed_item)
    
    # XML 데이터를 DataFrame으로 변환
    df = pd.DataFrame(items)
    
    # DataFrame을 CSV 파일로 저장 (Ensure UTF-8 encoding)
    df.to_csv("covid19_data_20220101.csv", index=False, encoding='utf-8-sig')
    
    # CSV 파일 저장 완료 메시지
    print("Data has been saved to covid19_data_20220101.csv")
else:
    print(f"Error {response.status_code}: {response.text}")


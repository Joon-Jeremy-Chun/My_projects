import requests
from bs4 import BeautifulSoup
from time import sleep
import pandas as pd

import time
import random
import datetime
import pymysql
from sqlalchemy import create_engine

pymysql.install_as_MySQLdb()
daily_engine = create_engine("", encoding='utf-8',pool_size = 100000,max_overflow = 0)

#   static variable
domain_url = "https://coinpan.com"
coin_dict = {
    "xrp": "xrp"}
crawling_count = 10000000
show_process_log = True
# show_process_log = False

for name, coin in coin_dict.items():
    if show_process_log: print(f"### {name} (전체 코인 중 ")
    if show_process_log: print(f">> 게시글 url 추출 시작")
    post_url_list = list()
    for page in range(0, 10):
        if show_process_log: print(f">> [{page} 페이지] 게시글 url 추출 (전체 중 {round(page*100/2, 2)}% 완료)")
        #Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36
        #Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) \Chrome/76.0.3809.146 Whale/2.6.89.9 Safari/537.36'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36',
                   'Referer': f"{domain_url}/index.php?mid={coin}&page={page-1 if page > 1 else page}"}
        res = requests.get(f"{domain_url}/index.php?mid={coin}&page={page}", headers=headers)
        page_soup = BeautifulSoup(res.text, "html.parser")
        time.sleep(0.07)
        for tr in page_soup.select("#board_list > table > tbody > tr")[1:]:
            if len(post_url_list) < crawling_count:
                post_url_list.append(tr.select_one("td.title a:nth-of-type(1)")["href"])
    if show_process_log: print(f">> {name}의 게시글 url 추출이 완료되었습니다. (총 {len(post_url_list)}개의 게시글 url)")
    time.sleep( random.uniform(2,4))

    if show_process_log: print(f">> 수집한 전체 게시글의 데이터 수집을 시작")
    title = [] 
    body = [] 
    date_time = []
    coin_name = []
    for i, url in enumerate(post_url_list):
        if show_process_log: print(f"- {domain_url + url} 게시글 크롤링 (전체 중 {round(i*100/len(post_url_list), 2)}% 완료)")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) \
                                              Chrome/76.0.3809.146 Whale/2.6.89.9 Safari/537.36',
                   'Referer': domain_url + url[:url.rindex("&")] if "&" in url else ""}
        post_soup = BeautifulSoup(requests.get(domain_url + url, headers=headers).text, "html.parser")
        target = post_soup.select_one('div.board_read.rd')
        title_t = target.select_one('div:nth-of-type(1) > div > div').text.strip()
        title.append(title_t)
        body_t = target.select_one('div.read_body > div').text.strip()
        body.append(body_t)
        date_time_t = target.select_one('div:nth-of-type(1) li:nth-of-type(6)').text.strip()
        date_time.append(date_time_t)
        coin_name.append(name)
        time.sleep( random.uniform(1,2))
   
        
    title_df = pd.DataFrame(title, columns=['title'])
    title_df = title_df.convert_dtypes()
    body_df = pd.DataFrame(body, columns=['body'])
    body_df = body_df.convert_dtypes()
    date_time_df = pd.DataFrame(date_time, columns=['datetime'])
    date_time_df = date_time_df.convert_dtypes()
    coin_name_df = pd.DataFrame(coin_name,columns=['coin_name'])
    df = pd.concat([title_df,body_df,date_time_df,coin_name_df],axis=1)
    time.sleep( random.uniform(1,4))

    print(f">> {name}의 데이터프레임 : ")
    daily_conn = daily_engine.connect()
    df.to_sql(name='coinpan_xrp',con=daily_conn,if_exists='append')
    print(name,'데이터 업데이트 완료\n')
    time.sleep( random.uniform(1,4))

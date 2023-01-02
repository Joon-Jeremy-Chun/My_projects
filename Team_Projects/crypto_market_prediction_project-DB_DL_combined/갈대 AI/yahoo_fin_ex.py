# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 12:12:19 2021

@author: joonc
"""
#%%
!pip install yahoo_fin
#%%
!pip install yahoo_fin --upgrade
#%%
import yahoo_fin.stock_info as si
#%%
# datetime
# feedparser
# ftplib
# io
# json
# pandas
# requests
# requests_html
#%%
!pip install datetime
!pip install feedparser
!pip install ftplib
!pip install io
!pip install json
!pip install pandas
!pip install requests
!pip install requests_html
#%%
from yahoo_fin.stock_info import get_analysts_info
from yahoo_fin.stock_info import get_balance_sheet
from yahoo_fin.stock_info import get_cash_flow
from yahoo_fin.stock_info import get_company_info
from yahoo_fin.stock_info import get_currencies
from yahoo_fin.stock_info import get_data
#%%
a = get_analysts_info('NFLX')
print(a)
#%%
b = get_analysts_info('MITC')
print(b)
#%%
c = get_analysts_info('AI')
print(c)
#%%
d = get_analysts_info('SNOW')
print(d)
#%%
e = get_analysts_info('RIVN')
print(e)
#%%
f = get_analysts_info('LCID')
print(f)
#%%
g = get_analysts_info('GOEV')
print(g)
#%%
h = get_analysts_info('BABA')
print(h)
#%%
i = get_analysts_info('MSFT')
print(i)
#%%
j = get_analysts_info('AAPL')
print(j)
#%%
k = get_analysts_info('NU')
print(k)
#%%
i_balance = get_balance_sheet('MSFT')
print(i_balance)
#%%
b_balance = get_balance_sheet('MITC')
print(b_balance)
#%%
b_cash_flow = get_cash_flow('MITC')
print(b_cash_flow)
#%%
c_cash_flow = get_cash_flow('AI')
print(c_cash_flow)
#%%
b_company_info = get_company_info('MITC')
print(b_company_info)
#%%
i_company_info = get_company_info('MSFT')
print(i_company_info)
#%%
curr = get_currencies()
print(curr)
#%%
i_price_data = get_data('MSFT', start_date = None , end_date = None ,index_as_date = True, interval = "1d") 
print(i_price_data)
#%%
i_price_data_2 = get_data('MSFT', start_date = '2000-01-02' , end_date = None ,index_as_date = True, interval = "1mo") 
print(i_price_data_2)
#%%
i_price_data_2.info()
type(i_price_data_2)
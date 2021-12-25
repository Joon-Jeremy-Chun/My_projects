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
#%%
a = get_analysts_info('nflx')
print(a)
#%%
b = get_analysts_info('mitc')
print(b)
#%%

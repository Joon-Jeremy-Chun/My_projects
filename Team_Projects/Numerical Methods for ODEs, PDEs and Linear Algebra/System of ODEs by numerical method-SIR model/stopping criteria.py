# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 17:57:06 2024

@author: Perez, Matthew
"""

T =[]

for i in range(0, 50):
    n = 10**(-i)
    T.append(n)

test = 0
#%%
while test < 100:
    for n in range(0, len(T)):
        new = 1 + T[n] 
        news = float(new)
        print(new)
        T[n] = news
    
    test_no = sum(T)
    test = float(test_no)
    print('T sum:', test_no)
    
print(test)
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 13:23:16 2023

@author: joonc
"""

#%%
#Use the given functions
#Simpson's Rule

#f;function, n; # of intervals, a;left end, b; right end
def Simf(f, n, a, b):
    iSum = 0
    h = abs(b-a)/n
    #print(iSum)
    
    for i in range(n+1):
        if i%2 != 0: #odd terms
            iSum += 4*f(i*(b-a)/n)
            #print('ith',i)
            #print('accumulate_i',iSum)
        if i%2 == 0: #even terms
            iSum += 2*f(i*(b-a)/n)
            #print('ith',i)
            #rint('accumulate_i',iSum)
            
    iSum += -f(a)
    iSum += -f(b)   
    #print(iSum)
    return h/3*iSum
#%%
#test
def f(x):
    return x*(1-x**2)
#%%
#test
print(Simf(f, 4, 0, 1))
#%%
#test2

import numpy as np

def g(x):
    return np.log(1+x) #in numpy log == ln
#%%
#test2

print(Simf(g, 100000, 0, 1))
#%%
print(Simf(g, 10, 0, 1))
#%%
print(Simf(g, 3, 0, 1))

#%%
#test3

def my_function(x):
   return 123.740 * x**0 + 3.475 * x**1 - 0.140 * x**2 + 0.001 * x**3
#%%
print(Simf(my_function, 100, 0, 90))
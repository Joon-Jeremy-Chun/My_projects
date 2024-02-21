# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 18:44:13 2024

@author: joonc
"""

# Compute the absolute error at each time step. Your output should include the print out of approximate value, exact value, and absolute error at each time step.
# Use your code to verify the order of accuracy of the Midpoint method
import numpy as np
import math
import matplotlib.pyplot as plt

#%%
#Define inline function
def model(t,y):
    dydt = (2/t)*y +(t**2)*np.exp(t)
    return dydt
#%%
#Define midpoint method
def Mid_ptn(f, y0, t):
    h = (b-a) / (len(t)-1)
    y =[]
    y.append(y0)
    y1 = y0+h*model(t[0],y0)
    y.append(y1)
    
    for i in t[1:-1]: 
        yn = y[-2] + 2*h*f(i,y[-1])
        y.append(yn)
    return y
#%%
#Define initial condition
y0 = 0
a = 1#starting
b = 2 #ending
#%%
#Define exact function y and exact value at b
def Exact_f(t):
    y = t**2*(np.exp(t)-np.exp(1))
    return y
Exact_y = Exact_f(b)
#%%
#Define N intervals (or lenghs)
Ns_inter = [2**i for i in np.linspace(1,17,17)] #2^17 intervals
#%% commpute the error and absolute error
approx_at_b = []
for n in Ns_inter:
    t = np.linspace(a,b, int(n) + 1) #N+1 points
    y = Mid_ptn(model, y0, t)
    y_b = y[-1]
    approx_at_b.append(y_b)
#%%
#compute approximate value, exact value, and absoulte error
abs_error = [abs(e - Exact_y) for e in approx_at_b] #absolute error
#%%
#print the output by approximate value, exact value, and absolute error at each time step
for a, c, b in zip(approx_at_b, abs_error, Ns_inter):
    print('approximate: ', "{:.15f}".format(a),', exact value: ', Exact_y, ', abs_error: ', "{:.15f}".format(c), ', N_intervals: ', b )

#%%
#Verify the order of accuracy by ratio of abs_error
abs_e_ratio = [ abs_error[i] / abs_error[i-1] for i in range(1,len(abs_error))]

#%%
#Verify by plot
#Covert into the natureal loge scales
log_values_e = [math.log(e) for e in abs_error]
log_value_n = [math.log(k) for k in Ns_inter]
#%%
x = [0,1,2,3,4,5,6,7,8,9,10,11,12]
y_1 = [ -1*x[i] for i in range(len(x))]
y_15 = [ -1.5*x[i] for i in range(len(x))]
y_2 = [ -2*x[i] for i in range(len(x))]
y_4 = [ -4*x[i] for i in range(len(x))]
y = [0,-2,-4,-6,-8,-10,-12,-14,-16,-18,-20,-22,-24]
plt.plot(log_value_n, log_values_e, 'v' ,label='abs_error in log')
plt.plot(x,y_1, label='test line: slope -1' )
plt.plot(x,y_15, label='test line: slope -1.5' )
plt.plot(x,y_2, label='test line: slope -2' )
plt.plot(x,y_4, label='test line: slope -4' )
plt.title('The patterns of abs_error, IVP')
plt.xlabel('Ns_tri (ln)')
plt.ylabel('abs_error (ln)')
plt.legend()
plt.grid(True)
plt.show()
#%%
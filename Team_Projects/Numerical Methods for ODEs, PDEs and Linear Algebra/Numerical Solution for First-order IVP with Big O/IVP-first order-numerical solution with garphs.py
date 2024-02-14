# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:20:34 2024

@author: joonc
"""
# Solve First-order IVP with Numerical method: Midpoint method.
# Consider the initial value problem:
#  y′ = 2/t*y +t^2*e^t, 1 ≤ t ≤ 2, y(1) = 0
#  with exact solution y(t) = t^2*(e^t − e).
#  The Midpoint method is given by
#  yn+1 = yn−1 +2hf(tn,yn).

#%%
import numpy as np
#from scipy.integrate import odeint
import matplotlib.pyplot as plt
#%%
#Define inline function
def model(t,y):
    dydt = (2/t)*y +(t**2)*np.exp(t)
    return dydt
#%%
#Define initial condition
y0 = 0
#%%
#Define time point
N=5 #intervals
a = 1 #starting
b = 2 #ending
t = np.linspace(a,b,N+1) #N+1 points
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
y = Mid_ptn(model, y0, t)
#%%
#Define exact function y
def Exact_f(t):
    y = t**2*(np.exp(t)-np.exp(1))
    return y
#%%
y_e = [ Exact_f(i) for i in t]
#%%
plt.plot(t, y , "v",label='midpoint method')
plt.plot(t, y_e, label='exact solution' )
plt.title(f'Intervals:{N} Exact solution vs midpoint method ')
plt.xlabel('T value')
plt.ylabel('y value')
plt.legend()
plt.grid(True)
plt.show()
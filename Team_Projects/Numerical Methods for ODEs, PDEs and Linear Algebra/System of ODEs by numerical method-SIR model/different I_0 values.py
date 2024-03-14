# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 19:02:38 2024

@author: joonc
"""
import numpy as np
import matplotlib.pyplot as plt
#%%
# create the matrix and plot for row1: Initial I0 values row2: percent of infected 
# Use for loop
# I0 values start 10^(0) to 10^(7) in every intervals of 10^(1/4)
I_inital = []
Percent_I =[]
log_t = []
for i in range(29):
    I_t = 10**(1/4*i)
    I_inital.append(I_t)
    
    r = 0.3 * 10**(-5)   # Infection rate
    a = 1.74  # Recovery rate

    def f_S(S, I, R, t):
        dSdt = -r*S*I
        return dSdt

    def f_I(S, I, R, t):
        dIdt = r*S*I - a*I
        return dIdt

    def f_R(S, I, R, t):
        dRdt = a*I
        return dRdt

    def trapezoidal_pc(y0, t0, fS, fI, fR, h, N):
        t = np.zeros(N+1)
        S = np.zeros(N+1)
        I = np.zeros(N+1)
        R = np.zeros(N+1)
        S[0], I[0], R[0] = y0
        t[0] = t0

        for n in range(N):
            # Predictor Step
            S_pred = S[n] + h * fS(S[n], I[n], R[n], t[n])
            I_pred = I[n] + h * fI(S[n], I[n], R[n], t[n])
            R_pred = R[n] + h * fR(S[n], I[n], R[n], t[n])

            # Corrector Step
            S[n+1] = S[n] + h * (fS(S[n], I[n], R[n], t[n]) + fS(S_pred, I_pred, R_pred, t[n+1])) / 2
            I[n+1] = I[n] + h * (fI(S[n], I[n], R[n], t[n]) + fI(S_pred, I_pred, R_pred, t[n+1])) / 2
            R[n+1] = R[n] + h * (fR(S[n], I[n], R[n], t[n]) + fR(S_pred, I_pred, R_pred, t[n+1])) / 2
            t[n+1] = t[n] + h

        return t, S, I, R
    
    t0 = 0
    h = 0.1
    N = 200

    S0 = 10**6
    I0 = I_t
    R0 = 0
    y0 = [S0, I0, R0]

    t, S, I, R = trapezoidal_pc(y0, t0, f_S, f_I, f_R, h, N)
   
    #S_long_term = S[-1]
    #I_long_term = I[-1]
    R_long_term = R[-1]

    print("Long-term values:")
    #print("Susceptible (S) =", S_long_term)
    #print("Infectious (I) =", I_long_term)
    #print("Recovered (R) =", R_long_term)
    total_p = S[0]+I[0]+R[0]
    total_infected = R[-1]/total_p
    Percent_I.append(total_infected)
    log_t.append(i)
    print("I0 value = ", I_t)
    print("Percent of infected =", total_infected*100)
#%%
#See the pattern in plot
plt.plot(log_t, Percent_I, label='Percent_I')
plt.xlabel('I_inital value (in 4*log(10))')
plt.ylabel('Percent_ifected people')
plt.title('I_0 values (in 10^(x/4)) and percent of Infected people')
plt.legend()
plt.grid(True)
plt.show()
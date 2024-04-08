# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 23:20:10 2024

@author: joonc
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# SIR model differential equations with two groups.
def deriv(y, t, Nc, Na, beta_cc, beta_ca, beta_ac, beta_aa, gamma_c, gamma_a, fg):
    Sc, Ic, Rc, Sa, Ia, Ra = y
    dScdt = -(beta_cc*Ic/Nc + beta_ac*Ia/Na)*Sc - fg*Sc 
    dIcdt = (beta_cc*Ic/Nc + beta_ac*Ia/Na)*Sc - gamma_c*Ic - fg*Ic
    dRcdt = gamma_c*Ic - fg*Rc
    dSadt = -(beta_aa*Ia/Na + beta_ca*Ic/Nc)*Sa + fg*Sc 
    dIadt = (beta_aa*Ia/Na + beta_ca*Ic/Nc)*Sa - gamma_a*Ia + fg*Ic 
    dRadt = gamma_a*Ia + fg*Rc 
    return dScdt, dIcdt, dRcdt, dSadt, dIadt, dRadt

# Initial number of individuals in each group
Ic0 = 0
Ia0 = 1
Rc0 = 0
Ra0 = 0
Sc0 = 150
Sa0 = 850
Nc = Sc0 + Ic0 + Rc0
Na = Sa0 + Ia0 + Ra0
#%%
#constants
# Contact rates, beta, and mean recovery rates, gamma, for children and adults (in 1/days)
beta_cc = 0.15
beta_ca = 0.3
gamma_c = 0.12
beta_aa = 0.35
beta_ac = 0.3
gamma_a = 0.1

# fg is age changes. It depence on only Children population.
fg = 0.001


#%%

# A grid of time points (in days)
t = np.linspace(0, 400, 4000)

# Initial conditions vector
y0 = Sc0, Ic0, Rc0, Sa0, Ia0, Ra0

# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(Nc, Na, beta_cc, beta_ca, beta_ac, beta_aa, gamma_c, gamma_a, fg))
Sc, Ic, Rc, Sa, Ia, Ra = ret.T

# Plot the data on six separate curves for Sc(t), Ic(t), Rc(t), Sa(t), Ia(t), and Ra(t)

plt.plot(t, Rc, label='Recovered Children')
plt.plot(t, Ra, label='Recovered Adults')
plt.xlabel('Days')
plt.ylabel('Population')
plt.title('SIR Model with Two Groups (Children and Adults)')
plt.legend()
plt.show()
#%%
# Plot the data on six separate curves for Sc(t), Ic(t), Rc(t), Sa(t), Ia(t), and Ra(t)
plt.plot(t, Sc, label='Susceptible Children')
plt.plot(t, Sa, label='Susceptible Adults')

plt.xlabel('Days')
plt.ylabel('Population')
plt.title('SIR Model with Two Groups (Children and Adults)')
plt.legend()
plt.show()
#%%
# Plot the data on six separate curves for Sc(t), Ic(t), Rc(t), Sa(t), Ia(t), and Ra(t)
plt.plot(t, Ic, label='Infected Children')
plt.plot(t, Ia, label='Infected Adults')

plt.xlabel('Days')
plt.ylabel('Population')
plt.title('SIR Model with Two Groups (Children and Adults)')
plt.legend()
plt.show()
#%%
# Plot the data on six separate curves for Sc(t), Ic(t), Rc(t), Sa(t), Ia(t), and Ra(t)
plt.plot(t, Sc, label='Susceptible Children')
plt.plot(t, Ic, label='Infected Children')
plt.plot(t, Rc, label='Recovered Children')
plt.plot(t, Sa, label='Susceptible Adults')
plt.plot(t, Ia, label='Infected Adults')
plt.plot(t, Ra, label='Recovered Adults')
plt.xlabel('Days')
plt.ylabel('Population')
plt.title('SIR Model with Two Groups (Children and Adults)')
plt.legend()
plt.show()
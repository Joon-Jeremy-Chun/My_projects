# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 10:04:15 2024

@author: joonc
"""


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# SIR model differential equations with two groups.
def deriv(y, t, Nc, Na, beta_cc, beta_ca, beta_ac, beta_aa, gamma_c, gamma_a, fg):
    Sc, Ic, Rc, Sa, Ia, Ra = y
    dScdt = -(beta_cc*Ic/Nc + beta_ac*Ia/Na)*Sc - fg*Sc + mu_A*Na
    dIcdt = (beta_cc*Ic/Nc + beta_ac*Ia/Na)*Sc - gamma_c*Ic - fg*Ic
    dRcdt = gamma_c*Ic - fg*Rc
    dSadt = -(beta_aa*Ia/Na + beta_ca*Ic/Nc)*Sa + fg*Sc - mu_A
    dIadt = (beta_aa*Ia/Na + beta_ca*Ic/Nc)*Sa - gamma_a*Ia + fg*Ic - mu_A
    dRadt = gamma_a*Ia + fg*Rc - mu_A
    return dScdt, dIcdt, dRcdt, dSadt, dIadt, dRadt

# Initial number of individuals in each group
Ic0 = 0
Ia0 = 1
Rc0 = 0
Ra0 = 0
Sc0 = 500
Sa0 = 9500
Nc = Sc0 + Ic0 + Rc0
Na = Sa0 + Ia0 + Ra0
#%%
#constants
# Contact rates, beta, and mean recovery rates, gamma, for children and adults (in 1/days)
beta_cc = 5.8
beta_ca = 1.3
gamma_c = .5
beta_aa = 2.6
beta_ac = 1.9
gamma_a = .5

# fg is age changes. It depence on only Children population.
fg = 0.001

#population birth and deat constant (assum total population  ==), population output apply only adult poplulation
mu_A = 0.0001
#%%

# A grid of time points (in days)
t = np.linspace(0, 20, 4000)

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

# Plot the data on six separate curves for Sc(t), Ic(t), Rc(t), Sa(t), Ia(t), and Ra(t)
plt.plot(t, Sc, label='Susceptible Children')
plt.plot(t, Sa, label='Susceptible Adults')

plt.xlabel('Days')
plt.ylabel('Population')
plt.title('SIR Model with Two Groups (Children and Adults)')
plt.legend()
plt.show()

# Plot the data on six separate curves for Sc(t), Ic(t), Rc(t), Sa(t), Ia(t), and Ra(t)
plt.plot(t, Ic, label='Infected Children')
plt.plot(t, Ia, label='Infected Adults')

plt.xlabel('Days')
plt.ylabel('Population')
plt.title('SIR Model with Two Groups (Children and Adults)')
plt.legend()
plt.show()

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
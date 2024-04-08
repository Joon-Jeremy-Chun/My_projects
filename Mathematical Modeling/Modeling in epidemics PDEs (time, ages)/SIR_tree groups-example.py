# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 08:28:12 2024

@author: joonc
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Extended SIR model differential equations including seniors.
def deriv(y, t, Nc, Na, Ns, beta_cc, beta_ca, beta_cs, beta_ac, beta_aa, beta_as, beta_sc, beta_sa, beta_ss, gamma_c, gamma_a, gamma_s, fg_ca, fg_as):
    Sc, Ic, Rc, Sa, Ia, Ra, Ss, Is, Rs = y
    dScdt = -(beta_cc*Ic/Nc + beta_ac*Ia/Na + beta_sc*Is/Ns)*Sc - fg_ca*Sc
    dIcdt = (beta_cc*Ic/Nc + beta_ac*Ia/Na + beta_sc*Is/Ns)*Sc - gamma_c*Ic - fg_ca*Ic
    dRcdt = gamma_c*Ic - fg_ca*Rc
    dSadt = -(beta_aa*Ia/Na + beta_ca*Ic/Nc + beta_sa*Is/Ns)*Sa + fg_ca*Sc - fg_as*Sa
    dIadt = (beta_aa*Ia/Na + beta_ca*Ic/Nc + beta_sa*Is/Ns)*Sa - gamma_a*Ia + fg_ca*Ic - fg_as*Ia
    dRadt = gamma_a*Ia + fg_ca*Rc - fg_as*Ra
    dSsdt = -(beta_ss*Is/Ns + beta_cs*Ic/Nc + beta_as*Ia/Na)*Ss + fg_as*Sa
    dIsdt = (beta_ss*Is/Ns + beta_cs*Ic/Nc + beta_as*Ia/Na)*Ss - gamma_s*Is + fg_as*Ia
    dRsdt = gamma_s*Is + fg_as*Ra
    return dScdt, dIcdt, dRcdt, dSadt, dIadt, dRadt, dSsdt, dIsdt, dRsdt

# Initial number of individuals in each group including seniors
Ic0, Ia0, Is0 = 1, 1, 1  # Initial number of infected individuals
Rc0, Ra0, Rs0 = 0, 0, 0  # Initial number of recovered individuals
Sc0, Sa0, Ss0 = 100, 800, 100  # Initial number of susceptible individuals
Nc, Na, Ns = Sc0 + Ic0 + Rc0, Sa0 + Ia0 + Ra0, Ss0 + Is0 + Rs0  # Total population in each group

# Model parameters
beta_cc, beta_ca, beta_cs = 0.15, 0.3, 0.2  # Transmission rates within and between children and other groups
beta_ac, beta_aa, beta_as = 0.3, 0.35, 0.15  # Transmission rates within and between adults and other groups
beta_sc, beta_sa, beta_ss = 0.2, 0.15, 0.25  # Transmission rates within and between seniors and other groups
gamma_c, gamma_a, gamma_s = 0.12, 0.1, 0.08  # Recovery rates for children, adults, and seniors
fg_ca, fg_as = 0.001, 0.0005  # Flow from children to adults and adults to seniors

# Time grid (in days)
t = np.linspace(0, 400, 4000)

# Initial conditions vector
y0 = Sc0, Ic0, Rc0, Sa0, Ia0, Ra0, Ss0, Is0, Rs0

# Integrate the extended SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(Nc, Na, Ns, beta_cc, beta_ca, beta_cs, beta_ac, beta_aa, beta_as, beta_sc, beta_sa, beta_ss, gamma_c, gamma_a, gamma_s, fg_ca, fg_as))
Sc, Ic, Rc, Sa, Ia, Ra, Ss, Is, Rs = ret.T

# Plotting
fig=plt.figure(figsize=(14, 10))

# Plot for children
plt.subplot(3,1,1)
plt.plot(t, Sc, 'b', label='Susceptible Children')
plt.plot(t, Ic, 'r', label='Infected Children')
plt.plot(t, Rc, 'g', label='Recovered Children')
plt.title('SIR Model Dynamics for Children, Adults, and Seniors')
plt.xlabel('Days')
plt.ylabel('Population')
plt.legend()

# Plot for adults
plt.subplot(3,1,2)
plt.plot(t, Sa, 'b', label='Susceptible Adults')
plt.plot(t, Ia, 'r', label='Infected Adults')
plt.plot(t, Ra, 'g', label='Recovered Adults')
plt.xlabel('Days')
plt.ylabel('Population')
plt.legend()

# Plot for seniors
plt.subplot(3,1,3)
plt.plot(t, Ss, 'b', label='Susceptible Seniors')
plt.plot(t, Is, 'r', label='Infected Seniors')
plt.plot(t, Rs, 'g', label='Recovered Seniors')
plt.xlabel('Days')
plt.ylabel('Population')
plt.legend()

plt.tight_layout()
plt.show()
#%%

# Save the figure
fig.savefig("SIR_model_2D_plots.png", dpi=600) 
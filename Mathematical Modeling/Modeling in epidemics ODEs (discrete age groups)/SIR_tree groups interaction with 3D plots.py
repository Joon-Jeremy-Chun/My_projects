# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 08:42:13 2024

@author: joonc
"""
#The model including group flow rate = fg (ageing)

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Adjusted group identifiers for zero-based indexing
CHILDREN, ADULTS, SENIORS = 0, 1, 2

# Extended SIR model differential equations including seniors.
def deriv(y, t, N, beta, gamma, fg):
    Sc, Ic, Rc, Sa, Ia, Ra, Ss, Is, Rs = y
    dScdt = -(beta[CHILDREN, CHILDREN] * Ic/N[CHILDREN] + beta[CHILDREN, ADULTS] * Ia/N[ADULTS] + beta[CHILDREN, SENIORS] * Is/N[SENIORS]) * Sc - fg[CHILDREN, ADULTS] * Sc
    dIcdt = (beta[CHILDREN, CHILDREN] * Ic/N[CHILDREN] + beta[CHILDREN, ADULTS] * Ia/N[ADULTS] + beta[CHILDREN, SENIORS] * Is/N[SENIORS]) * Sc - gamma[CHILDREN] * Ic - fg[CHILDREN, ADULTS] * Ic
    dRcdt = gamma[CHILDREN] * Ic - fg[CHILDREN, ADULTS] * Rc
    dSadt = -(beta[ADULTS, ADULTS] * Ia/N[ADULTS] + beta[ADULTS, CHILDREN] * Ic/N[CHILDREN] + beta[ADULTS, SENIORS] * Is/N[SENIORS]) * Sa + fg[CHILDREN, ADULTS] * Sc - fg[ADULTS, SENIORS] * Sa
    dIadt = (beta[ADULTS, ADULTS] * Ia/N[ADULTS] + beta[ADULTS, CHILDREN] * Ic/N[CHILDREN] + beta[ADULTS, SENIORS] * Is/N[SENIORS]) * Sa - gamma[ADULTS] * Ia + fg[CHILDREN, ADULTS] * Ic - fg[ADULTS, SENIORS] * Ia
    dRadt = gamma[ADULTS] * Ia + fg[CHILDREN, ADULTS] * Rc - fg[ADULTS, SENIORS] * Ra
    dSsdt = -(beta[SENIORS, SENIORS] * Is/N[SENIORS] + beta[SENIORS, CHILDREN] * Ic/N[CHILDREN] + beta[SENIORS, ADULTS] * Ia/N[ADULTS]) * Ss + fg[ADULTS, SENIORS] * Sa
    dIsdt = (beta[SENIORS, SENIORS] * Is/N[SENIORS] + beta[SENIORS, CHILDREN] * Ic/N[CHILDREN] + beta[SENIORS, ADULTS] * Ia/N[ADULTS]) * Ss - gamma[SENIORS] * Is + fg[ADULTS, SENIORS] * Ia
    dRsdt = gamma[SENIORS] * Is + fg[ADULTS, SENIORS] * Ra
    return dScdt, dIcdt, dRcdt, dSadt, dIadt, dRadt, dSsdt, dIsdt, dRsdt

# Initial conditions and parameters (unchanged)
group_initial_conditions = {
    CHILDREN: {'S0': 100, 'I0': 1, 'R0': 0},
    ADULTS: {'S0': 800, 'I0': 1, 'R0': 0},
    SENIORS: {'S0': 100, 'I0': 1, 'R0': 0}
}

# Beta matrix (unchanged)
beta = np.array([
    [0.15, 0.3, 0.2],  # CHILDREN to CHILDREN, ADULTS, SENIORS
    [0.3, 0.35, 0.15],  # ADULTS to CHILDREN, ADULTS, SENIORS
    [0.2, 0.15, 0.25]   # SENIORS to CHILDREN, ADULTS, SENIORS
])

# Recovery rates and flow rates (unchanged)
gamma = {CHILDREN: 0.12, ADULTS: 0.1, SENIORS: 0.08}
fg = {(CHILDREN, ADULTS): 0.001, (ADULTS, SENIORS): 0.0005}

# Total populations (unchanged)
N = {group: conditions['S0'] + conditions['I0'] + conditions['R0'] for group, conditions in group_initial_conditions.items()}

# Time grid (in days) and initial conditions vector (unchanged)
t = np.linspace(0, 400, 4000)
y0 = sum(([conditions[key] for key in ['S0', 'I0', 'R0']] for conditions in group_initial_conditions.values()), [])

# Integration and plotting (unchanged)
ret = odeint(deriv, y0, t, args=(N, beta, gamma, fg))
Sc, Ic, Rc, Sa, Ia, Ra, Ss, Is, Rs = ret.T
plt.figure(figsize=(14, 10))
for idx, group in enumerate([CHILDREN, ADULTS, SENIORS], start=1):
    plt.subplot(3, 1, idx)
    plt.plot(t, ret.T[3*(idx-1)], 'b', label=f'Susceptible {group}')
    plt.plot(t, ret.T[3*(idx-1)+1], 'r', label=f'Infected {group}')
    plt.plot(t, ret.T[3*(idx-1)+2], 'g', label=f'Recovered {group}')
    plt.title(f'SIR Model Dynamics for Group {group}')
    plt.xlabel('Days')
    plt.ylabel('Population')
    plt.legend()
plt.tight_layout()
plt.show()

#%%

# Create a new figure for the 3D plot
fig = plt.figure(figsize=(16, 12))

# 3D plot for Susceptible population
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(t, np.full_like(t, CHILDREN), Sc, label='Children', color='blue')
ax1.plot(t, np.full_like(t, ADULTS), Sa, label='Adults', color='red')
ax1.plot(t, np.full_like(t, SENIORS), Ss, label='Seniors', color='green')
ax1.set_title('Susceptible Populations')
ax1.set_xlabel('Days')
ax1.set_ylabel('Group')
ax1.set_zlabel('Population')
ax1.set_yticks([CHILDREN, ADULTS, SENIORS])
ax1.set_yticklabels(['Children', 'Adults', 'Seniors'])
ax1.legend()

# 3D plot for Infected population
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot(t, np.full_like(t, CHILDREN), Ic, label='Children', color='blue')
ax2.plot(t, np.full_like(t, ADULTS), Ia, label='Adults', color='red')
ax2.plot(t, np.full_like(t, SENIORS), Is, label='Seniors', color='green')
ax2.set_title('Infected Populations')
ax2.set_xlabel('Days')
ax2.set_ylabel('Group')
ax2.set_zlabel('Population')
ax2.set_yticks([CHILDREN, ADULTS, SENIORS])
ax2.set_yticklabels(['Children', 'Adults', 'Seniors'])
ax2.legend()

# 3D plot for Recovered population
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot(t, np.full_like(t, CHILDREN), Rc, label='Children', color='blue')
ax3.plot(t, np.full_like(t, ADULTS), Ra, label='Adults', color='red')
ax3.plot(t, np.full_like(t, SENIORS), Rs, label='Seniors', color='green')
ax3.set_title('Recovered Populations')
ax3.set_xlabel('Days')
ax3.set_ylabel('Group')
ax3.set_zlabel('Population')
ax3.set_yticks([CHILDREN, ADULTS, SENIORS])
ax3.set_yticklabels(['Children', 'Adults', 'Seniors'])
ax3.legend()

plt.tight_layout()
plt.show()
#%%

# Save the figure
fig.savefig("SIR_model_3D_plots.png", dpi=600) 
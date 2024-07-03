# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 21:39:05 2024

@author: joonc
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Group identifiers for zero-based indexing
HIGHLY_CAUTIOUS, MODERATELY_CAUTIOUS, LOW_CAUTIOUS = 0, 1, 2

def deriv(y, t, N, beta, gamma, delta, a1, a2, b1, b2):
    """
    Compute the derivatives for the SIR model with different caution levels, immunity loss, and group transitions.

    Parameters:
    y (list): List of S, I, TI values for each group.
    t (float): Time.
    N (list): Total population for each group.
    beta (list): Transmission rates for each group.
    gamma (float): Recovery rate for the infected group.
    delta (float): Rate at which temporarily immune individuals lose immunity and return to susceptible.
    a1, a2, b1, b2 (float): Transition rates between the groups.

    Returns:
    list: List of derivatives for S, I, TI values.
    """
    SH, SM, SL, I, TI = y

    total_N = sum(N)
    proportion_H = N[HIGHLY_CAUTIOUS] / total_N
    proportion_M = N[MODERATELY_CAUTIOUS] / total_N
    proportion_L = N[LOW_CAUTIOUS] / total_N

    dSHdt = -beta[HIGHLY_CAUTIOUS] * SH * I / N[HIGHLY_CAUTIOUS] + delta * TI * proportion_H \
            + b1 * SM - a1 * SH
    dSMdt = -beta[MODERATELY_CAUTIOUS] * SM * I / N[MODERATELY_CAUTIOUS] + delta * TI * proportion_M \
            + a1 * SH + b2 * SL - (a2 + b1) * SM
    dSLdt = -beta[LOW_CAUTIOUS] * SL * I / N[LOW_CAUTIOUS] + delta * TI * proportion_L \
            + a2 * SM - b2 * SL
    dIdt = (beta[HIGHLY_CAUTIOUS] * SH * I / N[HIGHLY_CAUTIOUS] +
            beta[MODERATELY_CAUTIOUS] * SM * I / N[MODERATELY_CAUTIOUS] +
            beta[LOW_CAUTIOUS] * SL * I / N[LOW_CAUTIOUS]) - gamma * I
    dTIdt = gamma * I - delta * TI

    return [dSHdt, dSMdt, dSLdt, dIdt, dTIdt]

# Initial conditions and parameters
initial_conditions = {
    HIGHLY_CAUTIOUS: {'S0': 800, 'I0': 10, 'TI0': 0},  # Increase initial infected to 10
    MODERATELY_CAUTIOUS: {'S0': 150, 'I0': 10, 'TI0': 0},  # Increase initial infected to 10
    LOW_CAUTIOUS: {'S0': 50, 'I0': 10, 'TI0': 0}  # Increase initial infected to 10
}

# Transmission rates for each group
beta = [0.25, 0.5, 0.75]

# Recovery rate
gamma = 0.1

# Immunity loss rate
delta = 0.00417  # Approximately corresponding to 8 months immunity duration

# Transition rates between groups
a1 = 0.2  # Caution Relaxation Rate (High to Moderate)
a2 = 0.2  # Caution Relaxation Rate (Moderate to Low)
b1 = 0.005  # Caution Intensification Rate (Moderate to High)
b2 = 0.005  # Caution Intensification Rate (Low to Moderate)

# Total populations
N = [initial_conditions[HIGHLY_CAUTIOUS]['S0'],
     initial_conditions[MODERATELY_CAUTIOUS]['S0'],
     initial_conditions[LOW_CAUTIOUS]['S0']]

# Initial conditions vector
y0 = [initial_conditions[HIGHLY_CAUTIOUS]['S0'], 
      initial_conditions[MODERATELY_CAUTIOUS]['S0'], 
      initial_conditions[LOW_CAUTIOUS]['S0'],
      initial_conditions[LOW_CAUTIOUS]['I0'], 0]  # SH, SM, SL, I, TI

# Time grid (in days)
t = np.linspace(0, 3000, 16000)  # Extended time grid for long-term behavior

# Integrate the SIR equations over the time grid
ret = odeint(deriv, y0, t, args=(N, beta, gamma, delta, a1, a2, b1, b2))

# Extract results
SH, SM, SL, I, TI = ret.T

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t, SH, 'b', label='Highly Cautious Susceptible')
plt.plot(t, SM, 'g', label='Moderately Cautious Susceptible')
plt.plot(t, SL, 'r', label='Low Cautious Susceptible')
plt.plot(t, I, 'y', label='Infected')
plt.plot(t, TI, 'k', label='Temporarily Immune')
plt.xlabel('Days')
plt.ylabel('Population')
plt.title('Awareness-Based SIR Model with Immunity Loss and Group Transitions')
plt.legend()
plt.grid()
plt.show()

# Check total population on a specific day
def find_closest_index(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

day = 51
day_index = find_closest_index(t, day)

total_population = SH[day_index] + SM[day_index] + SL[day_index] + I[day_index] + TI[day_index]

print(f"Total population on day {day}: {total_population}")


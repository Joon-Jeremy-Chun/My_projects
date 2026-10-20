# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 21:06:42 2025

@author: joonc

(SPORA)(3.0) GDP Loss Function Fitting

This script fits an exponential GDP-loss function L(m) of the form

    L(m) = a * exp(b * m) + c,

where:
    - m is the transmission-rate multiplier (0 < m ≤ 1),
      representing the effective contact reduction due to NPIs
      (e.g., social distancing Level I, II, III in Korea).
    - L(m) is the annualized GDP loss fraction (negative values),
      i.e., the fraction of annual GDP lost under a given NPI level.

Data source and interpretation
------------------------------
We use three (m, L(m)) data points that summarize the macroeconomic
impact of Korea's three social distancing levels during COVID-19:

    m = 0.70  →  L(m) = -0.002   (Level I, mild distancing)
    m = 0.35  →  L(m) = -0.018   (Level II, moderate distancing)
    m = 0.20  →  L(m) = -0.064   (Level III, stringent distancing)

These GDP loss fractions L(m) were adopted from Kim et al. (2022),
who estimated real GDP gaps using Bank of Korea data for each NPI level.
In our SPORA paper/thesis, we use the fitted function L(m) to
assign an economic cost to any intermediate multiplier m between 0 and 1.

Output
------
1. Prints the fitted parameters (a, b, c) to the console.
2. Prints the explicit fitted function:
       L(m) = a * exp(b * m) + c
3. Saves a figure showing:
       - the three original data points, and
       - the fitted exponential curve
   to "Figures/GDP_loss_function.png".

@author: joonc
Created on Wed Mar 26 19:08:00 2025
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ------------------------------------------------------------
# 1. Define the exponential GDP loss function
# ------------------------------------------------------------
def gdp_loss(m, a, b, c):
    """
    Exponential GDP loss function.

    Parameters
    ----------
    m : float or array-like
        Transmission-rate multiplier (0 < m ≤ 1).
    a, b, c : floats
        Model parameters to be fitted.

    Returns
    -------
    L : float or array-like
        Annual GDP loss fraction L(m).
    """
    return a * np.exp(b * m) + c


# ------------------------------------------------------------
# 2. Given (m, L(m)) data points from macroeconomic analysis
# ------------------------------------------------------------
# m-values: phase-averaged transmission multipliers for
#           Korea's social distancing levels I, II, III
x_data = np.array([0.70, 0.35, 0.20])

# L(m): annualized GDP loss fraction (negative = GDP contraction)
#       adopted from Kim et al. (2022) / Bank of Korea analysis.
y_data = np.array([-0.002, -0.018, -0.064])


# ------------------------------------------------------------
# 3. Fit parameters (a, b, c) using nonlinear least squares
# ------------------------------------------------------------
# Initial guess for [a, b, c]; signs chosen to encourage a
# decreasing (more negative) loss as m decreases.
initial_guess = [-1.0, -1.0, -1.0]

# Increase the maximum number of function evaluations for robustness
popt, pcov = curve_fit(
    gdp_loss,
    x_data,
    y_data,
    p0=initial_guess,
    maxfev=2000
)
a, b, c = popt

print("Fitted parameters:")
print("a = {:.4f}, b = {:.4f}, c = {:.4f}".format(a, b, c))

# Print the fitted function in a readable form
print("Fitted GDP loss function:")
print("L(m) = {:.4f} * exp({:.4f} * m) + {:.4f}".format(a, b, c))


# ------------------------------------------------------------
# 4. Plot original data and fitted exponential curve
# ------------------------------------------------------------
# Generate a smooth range of m-values for plotting
x_fit = np.linspace(min(x_data) - 0.1, max(x_data) + 0.1, 100)
y_fit = gdp_loss(x_fit, a, b, c)

plt.figure(figsize=(8, 5))

# Original data points
plt.scatter(
    x_data,
    y_data,
    color="red",
    label="Data points (NPI levels I–III)"
)

# Fitted curve
plt.plot(
    x_fit,
    y_fit,
    color="blue",
    label="Exponential fit: L(m) = a·exp(b·m)+c"
)

plt.xlabel("Transmission multiplier  m")
plt.ylabel("Annual GDP loss fraction  L(m)")
plt.title("Fitted GDP Loss Function  L(m)")
plt.legend()
plt.grid(True)

# ------------------------------------------------------------
# 5. Save figure to 'Figures' folder
# ------------------------------------------------------------
os.makedirs("Figures", exist_ok=True)
plt.savefig("Figures/GDP_loss_function.png", dpi=300, bbox_inches="tight")

plt.show()

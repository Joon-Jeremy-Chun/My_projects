"""
(SPORA)(1.0) Beta matrix parameter fitting with SIRVD model

Created on Fri Nov 21 16:08:00 2025

@author: joonc

This script:
  1. Loads age-structured SIRV(D) data for Korea.
  2. Fits a 3x3 transmission matrix beta using L-BFGS-B
     over the February 2020 window.
  3. Solves the SIRVD model with the fitted beta.
  4. Produces age-specific and total I(t) plots.
  5. Saves the fitted beta matrix to a SPORA-specific CSV file.

NOTE:
  - A death compartment D_i and mortality rates mu_i are included
    in the ODE system, but for baseline beta fitting we set mu_i = 0.
    You can later replace mu with age-specific mortality rates if desired.
"""

import os
import pandas as pd
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 0. Global plotting style
# ----------------------------------------------------------------------
plt.rcParams.update({
    "figure.figsize"  : (6.4, 4.8),   # default literal size in inches
    "savefig.dpi"     : 300,          # high-resolution output
    "figure.dpi"      : 150,          # screen / notebook preview
    "font.size"       : 30,           # main text in each panel
    "axes.titlesize"  : 25,
    "axes.labelsize"  : 20,
    "xtick.labelsize" : 12,
    "ytick.labelsize" : 15,
    "legend.fontsize" : 17,
    "lines.linewidth" : 2.5,
    "pdf.fonttype"    : 42,           # embed fonts (good for LaTeX)
    "ps.fonttype"     : 42
})

# ----------------------------------------------------------------------
# 1. Paths & output folders
# ----------------------------------------------------------------------
fig_dir = '(SPORA)figures'   # SPORA-specific figure folder
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

file_path_parameters = 'DataSets/Korea_threeGroups_SIRV_parameter.csv'
file_path_inputs = 'Inputs.xlsx'

# ----------------------------------------------------------------------
# 2. Load data and fixed parameters
# ----------------------------------------------------------------------
# Real-world data
real_data = pd.read_csv(file_path_parameters)
real_data['Date'] = pd.to_datetime(real_data['Date'])

# Fixed parameters from Inputs.xlsx
inputs = pd.read_excel(file_path_inputs, header=None)

initial_beta = inputs.iloc[0:3, 0:3].values
recovery_rates = inputs.iloc[4, 0:3].values
vaccination_rates = inputs.iloc[10, 0:3].values
waning_immunity_rate = inputs.iloc[8, 0]

# Age groups & parameters
age_groups = ['0-19', '20-49', '50-80+']
gamma = recovery_rates
a = vaccination_rates
delta = waning_immunity_rate

# Disease-induced mortality rates mu_i.
# For baseline beta fitting we set these to zero.
# You can replace this with empirical age-specific values if desired.
# In 2020.2 there are 16 Death. (It is small number of poeple from 50,000,000)
mu = np.zeros(3)
# mu = (0.0001,0.0001,0.0001)

# ----------------------------------------------------------------------
# 3. Select fitting window (February 2020)
# ----------------------------------------------------------------------
start_date = '2020-02-01'   # Start date
end_date = '2020-03-01'     # End date (exclusive)

fit_data = real_data[(real_data['Date'] >= start_date) &
                     (real_data['Date'] <= end_date)].reset_index(drop=True)

if fit_data.empty:
    raise ValueError("No data found between the specified start_date and "
                     "end_date. Please check the date range.")

# ----------------------------------------------------------------------
# 4. SIRVD model ODE system
# ----------------------------------------------------------------------
def sirvd_model(y, t, beta, gamma, a, delta, mu):
    """
    Age-structured SIRVD model with 3 age groups.

    State vector y = [
        S1, I1, R1, V1, D1,
        S2, I2, R2, V2, D2,
        S3, I3, R3, V3, D3
    ]
    """
    (S1, I1, R1, V1, D1,
     S2, I2, R2, V2, D2,
     S3, I3, R3, V3, D3) = y

    # Live population in each age group (exclude D_i from denominator)
    N1 = S1 + I1 + R1 + V1
    N2 = S2 + I2 + R2 + V2
    N3 = S3 + I3 + R3 + V3
    N = [N1, N2, N3]

    # Force of infection λ_i
    I_vec = [I1, I2, I3]
    lambda1 = sum(beta[0, j] * Ij / Nj for j, Ij, Nj in zip(range(3), I_vec, N))
    lambda2 = sum(beta[1, j] * Ij / Nj for j, Ij, Nj in zip(range(3), I_vec, N))
    lambda3 = sum(beta[2, j] * Ij / Nj for j, Ij, Nj in zip(range(3), I_vec, N))

    # Group 1
    dS1 = -lambda1 * S1 - a[0] * S1 + delta * (R1 + V1)
    dI1 = lambda1 * S1 - gamma[0] * I1 - mu[0] * I1
    dR1 = gamma[0] * I1 - delta * R1
    dV1 = a[0] * S1 - delta * V1
    dD1 = mu[0] * I1

    # Group 2
    dS2 = -lambda2 * S2 - a[1] * S2 + delta * (R2 + V2)
    dI2 = lambda2 * S2 - gamma[1] * I2 - mu[1] * I2
    dR2 = gamma[1] * I2 - delta * R2
    dV2 = a[1] * S2 - delta * V2
    dD2 = mu[1] * I2

    # Group 3
    dS3 = -lambda3 * S3 - a[2] * S3 + delta * (R3 + V3)
    dI3 = lambda3 * S3 - gamma[2] * I3 - mu[2] * I3
    dR3 = gamma[2] * I3 - delta * R3
    dV3 = a[2] * S3 - delta * V3
    dD3 = mu[2] * I3

    return [dS1, dI1, dR1, dV1, dD1,
            dS2, dI2, dR2, dV2, dD2,
            dS3, dI3, dR3, dV3, dD3]

# ----------------------------------------------------------------------
# 5. Objective function for beta fitting
# ----------------------------------------------------------------------
def objective(beta_flat):
    """
    Mean squared error between modelled I_i(t) and observed I_i(t)
    over the fitting window.
    """
    beta = beta_flat.reshape((3, 3))

    # Initial conditions from data (D_i(0) assumed 0)
    y0 = [
        fit_data['S_0-19'][0], fit_data['I_0-19'][0],
        fit_data['R_0-19'][0], fit_data['V_0-19'][0], 0.0,
        fit_data['S_20-49'][0], fit_data['I_20-49'][0],
        fit_data['R_20-49'][0], fit_data['V_20-49'][0], 0.0,
        fit_data['S_50-80+'][0], fit_data['I_50-80+'][0],
        fit_data['R_50-80+'][0], fit_data['V_50-80+'][0], 0.0
    ]

    t = range(len(fit_data))

    # Solve ODEs
    result = odeint(sirvd_model, y0, t, args=(beta, gamma, a, delta, mu))

    # Infectious indices in the 15-dimensional state vector:
    # I1 -> index 1, I2 -> index 6, I3 -> index 11
    I_model = result[:, [1, 6, 11]]

    # Observed infectious data
    I_real = fit_data[['I_0-19', 'I_20-49', 'I_50-80+']].values

    error = np.mean((I_model - I_real) ** 2)
    return error

# ----------------------------------------------------------------------
# 6. Optimize transmission matrix beta
# ----------------------------------------------------------------------
bounds = [(0, None)] * 9  # beta_ij >= 0

result = minimize(objective,
                  initial_beta.flatten(),
                  method='L-BFGS-B',
                  bounds=bounds)

if result.success:
    optimal_beta = result.x.reshape((3, 3))
    print("Optimization Successful!")
    print(optimal_beta)
else:
    print("Optimization Failed:", result.message)
    raise SystemExit

# ----------------------------------------------------------------------
# 7. Solve model with fitted beta & create plots
# ----------------------------------------------------------------------
y0 = [
    fit_data['S_0-19'][0], fit_data['I_0-19'][0],
    fit_data['R_0-19'][0], fit_data['V_0-19'][0], 0.0,
    fit_data['S_20-49'][0], fit_data['I_20-49'][0],
    fit_data['R_20-49'][0], fit_data['V_20-49'][0], 0.0,
    fit_data['S_50-80+'][0], fit_data['I_50-80+'][0],
    fit_data['R_50-80+'][0], fit_data['V_50-80+'][0], 0.0
]
t = range(len(fit_data))

fitted_result = odeint(sirvd_model, y0, t,
                       args=(optimal_beta, gamma, a, delta, mu))

# Infectious components for plotting
I_fitted = fitted_result[:, [1, 6, 11]]

title_map = {'0-19': '0–19',
             '20-49': '20–49',
             '50-80+': '50+'}

# --- 7A. Individual age-group panels -----------------------------------
for idx, age_group in enumerate(age_groups):
    plt.figure(figsize=(10, 6))
    plt.plot(fit_data['Date'],
             fit_data[f'I_{age_group}'],
             label='Actual I(t)', linestyle='--')
    plt.plot(fit_data['Date'],
             I_fitted[:, idx],
             label='Fitted I(t)', color='red')

    plt.title(f"Actual vs Fitted I(t)  —  Age {title_map[age_group]}")
    plt.xlabel("Date")
    plt.ylabel("Infectious population")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # ★ Added: hide last x-tick label (remove "Mar 01")
    ax = plt.gca()
    labels = [label.get_text() for label in ax.get_xticklabels()]
    if len(labels) > 0:
        labels[-1] = ""     # remove last tick label
        ax.set_xticklabels(labels)

    save_path = os.path.join(
        fig_dir,
        f"Actual_vs_Fitted_I_{age_group.replace('-', '_')}.png"
    )
    plt.savefig(save_path, dpi=300)
    plt.show()

# --- 7B. Total infectious population panel ------------------------------
total_actual_I = (fit_data['I_0-19'] +
                  fit_data['I_20-49'] +
                  fit_data['I_50-80+'])

total_fitted_I = I_fitted.sum(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(fit_data['Date'], total_actual_I,
         label='Actual total I(t)', linestyle='--')
plt.plot(fit_data['Date'], total_fitted_I,
         label='Fitted total I(t)', color='red')

plt.title("Actual vs Fitted  —  Total infectious population")
plt.xlabel("Date")
plt.ylabel("Infectious population")
plt.legend()
plt.grid(True)
plt.tight_layout()

# ★ Added: hide last x-tick label (remove "Mar 01")
ax = plt.gca()
labels = [label.get_text() for label in ax.get_xticklabels()]
if len(labels) > 0:
    labels[-1] = ""
    ax.set_xticklabels(labels)

save_path_total = os.path.join(fig_dir, "Actual_vs_Fitted_Total_I.png")
plt.savefig(save_path_total, dpi=300)
plt.show()


# ----------------------------------------------------------------------
# 8. Save fitted beta matrix (SPORA-specific name)
# ----------------------------------------------------------------------
output_file = 'DataSets/Fitted_Beta_Matrix_SPORA.csv'
pd.DataFrame(optimal_beta,
             columns=age_groups,
             index=age_groups).to_csv(output_file, index=True)

print("Fitted β-matrix saved to:", output_file)

# -*- coding: utf-8 -*-
"""
Full Script: SIRV Simulation with Corrected Vaccination Flows
-------------------------------------------------------------
Loads parameters, defines corrected ODE, and runs three scenarios:
  1) Baseline: no distancing, no vaccination
  2) Seniors-only vaccination @ βₘ=0.6
  3) Adults-only vaccination  @ βₘ=0.6
Prints peak infected and peak day for each.
"""

import os
import pandas as pd
import numpy as np
from scipy.integrate import odeint

# Create output directory
if not os.path.exists("Figures"):
    os.makedirs("Figures")

# 1. Load parameters from Excel and CSV
def load_parameters(file_path):
    df = pd.read_excel(file_path, header=None)
    # Recovery rates (3 values)
    recovery_rates      = df.iloc[4, 0:3].values
    # Maturity rates (2 values)
    maturity_rates      = df.iloc[6, 0:2].values
    # Waning immunity rate (1 value)
    waning_immunity_rate= df.iloc[8, 0]
    # Vaccination rates (3 values, not used here)
    vaccination_rates   = df.iloc[10, 0:3].values
    # Time span (int days)
    time_span           = int(df.iloc[12, 0])
    # Population sizes (3 values)
    population_size     = df.iloc[14, 0:3].values
    # Initial compartments
    susceptible_init    = df.iloc[14, 0:3].values
    infectious_init     = df.iloc[16, 0:3].values
    recovered_init      = df.iloc[18, 0:3].values
    vaccinated_init     = df.iloc[20, 0:3].values
    return (recovery_rates, maturity_rates, waning_immunity_rate,
            vaccination_rates, time_span, population_size,
            susceptible_init, infectious_init, recovered_init, vaccinated_init)

# Paths
file_path = 'Inputs.xlsx'
beta_csv  = 'DataSets/Fitted_Beta_Matrix.csv'

# Load
gamma, mu, W, a_rates, time_span, N, S_init, I_init, R_init, V_init = load_parameters(file_path)
beta = pd.read_csv(beta_csv).iloc[0:3, 1:4].values

# Initial conditions and time vector
initial_conditions = [
    S_init[0], I_init[0], R_init[0], V_init[0],
    S_init[1], I_init[1], R_init[1], V_init[1],
    S_init[2], I_init[2], R_init[2], V_init[2]
]
t = np.linspace(0, time_span, time_span)

# 2. Corrected ODE: vacc_rates are per-day fractions (not divided by N inside)
def deriv(y, t, N, beta, gamma, mu, W, vacc_rates):
    S1,I1,R1,V1, S2,I2,R2,V2, S3,I3,R3,V3 = y
    # Force of infection
    lam1 = beta[0,0]*I1/N[0] + beta[0,1]*I2/N[1] + beta[0,2]*I3/N[2]
    lam2 = beta[1,0]*I1/N[0] + beta[1,1]*I2/N[1] + beta[1,2]*I3/N[2]
    lam3 = beta[2,0]*I1/N[0] + beta[2,1]*I2/N[1] + beta[2,2]*I3/N[2]
    # Equations with corrected vaccination flow
    dS1 = -lam1*S1 - vacc_rates[0]*S1 + W*(R1+V1)
    dI1 =  lam1*S1 - gamma[0]*I1 - mu[0]*I1
    dR1 =  gamma[0]*I1 - W*R1 - mu[0]*R1
    dV1 =  vacc_rates[0]*S1 - W*V1

    dS2 = -lam2*S2 - vacc_rates[1]*S2 + W*(R2+V2) + mu[0]*S1
    dI2 =  lam2*S2 - gamma[1]*I2 - mu[1]*I2 + mu[0]*I1
    dR2 =  gamma[1]*I2 - W*R2 - mu[1]*R2 + mu[0]*R1
    dV2 =  vacc_rates[1]*S2 - W*V2

    dS3 = -lam3*S3 - vacc_rates[2]*S3 + W*(R3+V3) + mu[1]*S2
    dI3 =  lam3*S3 - gamma[2]*I3 + mu[1]*I2
    dR3 =  gamma[2]*I3 - W*R3 + mu[1]*R2
    dV3 =  vacc_rates[2]*S3 - W*V3

    return (dS1,dI1,dR1,dV1, dS2,dI2,dR2,dV2, dS3,dI3,dR3,dV3)

# Compute daily global vaccination capacity fraction
daily_capacity_frac = 0.0107  # 1.07% of total pop per day

# Scenario definitions
scenarios = [
    ("Baseline: no vaccine/no SD", 1.0, 0.0, 0.0, 0.0),
    ("Seniors-only @ βₘ=0.6",      0.6, 0.0, 0.0, 0.0396),
    ("Adults-only @ βₘ=0.6",      0.6, 0.0, 
        daily_capacity_frac * (N.sum()/N[1]), 0.0),
]

# Run scenarios
for name, beta_m, v1, v2, v3 in scenarios:
    beta_mod     = beta * beta_m
    vacc_rates   = np.array([v1, v2, v3])
    results      = odeint(deriv, initial_conditions, t,
                          args=(N, beta_mod, gamma, mu, W, vacc_rates))
    I_total      = results[:,1] + results[:,5] + results[:,9]
    peak_value   = I_total.max()
    peak_day     = t[np.argmax(I_total)]

    print(f"=== {name} ===")
    print(f"βₘ                = {beta_m:.2f}")
    print(f"Vaccination rates = v1={v1:.4f}, v2={v2:.4f}, v3={v3:.4f}")
    print(f"Peak total I      = {peak_value:.0f}")
    print(f"Day of peak       = {peak_day:.1f}\n")


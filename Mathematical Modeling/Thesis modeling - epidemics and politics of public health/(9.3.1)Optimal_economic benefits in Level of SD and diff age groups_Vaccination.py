# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 19:06:00 2025

@author: joonc
"""

# -*- coding: utf-8 -*-
"""
Combined Example: Optimization Printout & Multiple Heat Maps
-------------------------------------------------------------
This script runs a 3-group SIRV model simulation and computes day‐by‐day costs.
It then:
  1) Loops over (beta_m, v1, v2) to determine the best (minimum cost) vaccination scenario per beta_m,
     prints a summary table, and saves the results to CSV.
  2) Builds 2D cost arrays for each beta_m and plots heat maps of the final cost vs. (v1, v2).
  
Requirements: numpy, pandas, matplotlib, scipy.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# (OPTIONAL) Create output folder if it does not exist
if not os.path.exists("Figures"):
    os.makedirs("Figures")

# ----------------------------------------------------------------
# 1. Population & Disease Model Parameters
# ----------------------------------------------------------------
# Age groups: children (0–19), adults (20–49), seniors (50+)
N = np.array([6246073, 31530507, 13972160])  # [N_children, N_adults, N_seniors]
S_init = N.copy()          # all susceptible
I_init = np.array([1, 1, 1]) # one infected in each group
R_init = np.array([0, 0, 0]) # none recovered
V_init = np.array([0, 0, 0]) # none vaccinated

# Initial conditions for S, I, R, V in each group (total length = 12)
initial_conditions = np.array([
    S_init[0], I_init[0], R_init[0], V_init[0],
    S_init[1], I_init[1], R_init[1], V_init[1],
    S_init[2], I_init[2], R_init[2], V_init[2]
])

gamma = np.array([0.125, 0.083, 0.048])  # recovery rates for children, adults, seniors
W = 0.005                                 # waning immunity rate (applies to R and V)

# Baseline transmission matrix
beta_base = np.array([
    [0.1901, 0.0660, 0.0028],
    [0.1084, 0.1550, 0.0453],
    [0.1356, 0.1540, 0.1165]
])

# Daily vaccination capacity (about 1.07% of entire population)
total_pop = np.sum(N)
daily_vacc_capacity = 0.0107 * total_pop

# Time settings for ODE solve (e.g., 150 days)
time_span = 150
t = np.linspace(0, time_span, time_span)  # integer days 0,1,...,149

# ----------------------------------------------------------------
# 2. Economic Parameters
# ----------------------------------------------------------------
# Medical costs
cost_mild = 160.8
cost_hosp = 432.5
cost_icu  = 1129.5

# Fractions for mild, hospitalized, and ICU cases
f_child_mild  = 0.95
f_child_hosp  = 0.04
f_child_icu   = 0.01

f_adult_mild  = 0.90
f_adult_hosp  = 0.08
f_adult_icu   = 0.02

f_senior_mild = 0.70
f_senior_hosp = 0.20
f_senior_icu  = 0.10

# Wage, funeral, and mortality parameters
employment_rates = np.array([0.02, 0.694, 0.513])
daily_income     = np.array([27.74, 105.12, 92.45])
funeral_cost     = 9405.38
mortality_fraction = np.array([0.000001, 0.00003, 0.007])

# Vaccination cost per capita
vaccination_cost_per_capita = 35.76

# GDP parameters
GDP_per_capita = 31929
a, b, c = -0.3648, -8.7989, -0.0012  # GDP loss model: fraction = a*exp(b*beta_m)+c

# ----------------------------------------------------------------
# 3. ODE and Helper Functions
# ----------------------------------------------------------------
def deriv_sirv(y, t, N, beta_mod, gamma, W, v1, v2, v3):
    """
    Computes the derivatives for the 3-group SIRV model.
    y: 12-element vector [S1,I1,R1,V1, S2,I2,R2,V2, S3,I3,R3,V3]
    """
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y

    # Forces of infection for each group
    lambda1 = beta_mod[0,0]*I1/N[0] + beta_mod[0,1]*I2/N[1] + beta_mod[0,2]*I3/N[2]
    lambda2 = beta_mod[1,0]*I1/N[0] + beta_mod[1,1]*I2/N[1] + beta_mod[1,2]*I3/N[2]
    lambda3 = beta_mod[2,0]*I1/N[0] + beta_mod[2,1]*I2/N[1] + beta_mod[2,2]*I3/N[2]

    # Children
    dS1dt = -lambda1 * S1 - v1 * S1 + W * (R1 + V1)
    dI1dt = lambda1 * S1 - gamma[0] * I1
    dR1dt = gamma[0] * I1 - W * R1
    dV1dt = v1 * S1 - W * V1

    # Adults
    dS2dt = -lambda2 * S2 - v2 * S2 + W * (R2 + V2)
    dI2dt = lambda2 * S2 - gamma[1] * I2
    dR2dt = gamma[1] * I2 - W * R2
    dV2dt = v2 * S2 - W * V2

    # Seniors
    dS3dt = -lambda3 * S3 - v3 * S3 + W * (R3 + V3)
    dI3dt = lambda3 * S3 - gamma[2] * I3
    dR3dt = gamma[2] * I3 - W * R3
    dV3dt = v3 * S3 - W * V3

    return [dS1dt, dI1dt, dR1dt, dV1dt,
            dS2dt, dI2dt, dR2dt, dV2dt,
            dS3dt, dI3dt, dR3dt, dV3dt]

def compute_v3(v1, v2, N, daily_vacc_capacity):
    """
    Given v1 and v2 for children and adults,
    computes v3 for seniors so that the total number vaccinated equals the daily capacity.
    """
    numer = daily_vacc_capacity - (N[0]*v1 + N[1]*v2)
    return numer / N[2]

def compute_cost_dataframe(results, beta_m, N, dt=1.0):
    """
    Computes day-by-day and cumulative costs over the simulation.
    Returns a DataFrame containing compartment sizes and various cost components.
    """
    T = results.shape[0]
    times = np.arange(T) * dt

    med_arr   = np.zeros(T)
    wage_arr  = np.zeros(T)
    death_arr = np.zeros(T)
    vacc_arr  = np.zeros(T)
    gdp_arr   = np.zeros(T)
    total_arr = np.zeros(T)

    cum_med_arr   = np.zeros(T)
    cum_wage_arr  = np.zeros(T)
    cum_death_arr = np.zeros(T)
    cum_vacc_arr  = np.zeros(T)
    cum_gdp_arr   = np.zeros(T)
    cum_total_arr = np.zeros(T)

    # GDP daily cost (loss rate)
    GDP_loss_fraction = a * np.exp(b * beta_m) + c
    GDP_loss_rate = GDP_per_capita * abs(GDP_loss_fraction) * (np.sum(N) / 365.0)

    for i in range(T):
        S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = results[i]

        # Infected counts by group
        I_child  = I1
        I_adult  = I2
        I_senior = I3

        # Medical costs
        med_child  = I_child * (f_child_mild * cost_mild + f_child_hosp * cost_hosp + f_child_icu * cost_icu)
        med_adult  = I_adult * (f_adult_mild * cost_mild + f_adult_hosp * cost_hosp + f_adult_icu * cost_icu)
        med_senior = I_senior * (f_senior_mild * cost_mild + f_senior_hosp * cost_hosp + f_senior_icu * cost_icu)
        med = med_child + med_adult + med_senior

        # Wage loss costs
        w_child  = daily_income[0] * employment_rates[0] * I_child
        w_care   = daily_income[1] * employment_rates[1] * I_child  # caregiver for sick children
        w_adult  = daily_income[1] * employment_rates[1] * I_adult
        w_senior = daily_income[2] * employment_rates[2] * I_senior
        wage = w_child + w_care + w_adult + w_senior

        # Funeral (death) costs
        death = funeral_cost * (
            mortality_fraction[0] * I_child +
            mortality_fraction[1] * I_adult +
            mortality_fraction[2] * I_senior
        )

        # Vaccination costs
        vacc = vaccination_cost_per_capita * (V1 + V2 + V3)

        # Daily GDP loss
        gdp_daily = GDP_loss_rate

        total = med + wage + death + vacc + gdp_daily

        med_arr[i]   = med
        wage_arr[i]  = wage
        death_arr[i] = death
        vacc_arr[i]  = vacc
        gdp_arr[i]   = gdp_daily
        total_arr[i] = total

        if i == 0:
            cum_med_arr[i]   = med
            cum_wage_arr[i]  = wage
            cum_death_arr[i] = death
            cum_vacc_arr[i]  = vacc
            cum_gdp_arr[i]   = gdp_daily
            cum_total_arr[i] = total
        else:
            cum_med_arr[i]   = cum_med_arr[i-1]   + med
            cum_wage_arr[i]  = cum_wage_arr[i-1]  + wage
            cum_death_arr[i] = cum_death_arr[i-1] + death
            cum_vacc_arr[i]  = cum_vacc_arr[i-1]  + vacc
            cum_gdp_arr[i]   = cum_gdp_arr[i-1]   + gdp_daily
            cum_total_arr[i] = cum_total_arr[i-1] + total

    df = pd.DataFrame({
        "time": times,
        "S1": results[:, 0],
        "I1": results[:, 1],
        "R1": results[:, 2],
        "V1": results[:, 3],
        "S2": results[:, 4],
        "I2": results[:, 5],
        "R2": results[:, 6],
        "V2": results[:, 7],
        "S3": results[:, 8],
        "I3": results[:, 9],
        "R3": results[:, 10],
        "V3": results[:, 11],
        "Med_Cost": med_arr,
        "Wage_Loss": wage_arr,
        "Death_Cost": death_arr,
        "Vacc_Cost": vacc_arr,
        "GDP_Loss": gdp_arr,
        "Daily_Total": total_arr,
        "Cumulative_Med_Cost": cum_med_arr,
        "Cumulative_Wage_Loss": cum_wage_arr,
        "Cumulative_Death_Cost": cum_death_arr,
        "Cumulative_Vacc_Cost": cum_vacc_arr,
        "Cumulative_GDP_Loss": cum_gdp_arr,
        "Cumulative_Cost": cum_total_arr
    })
    return df

def run_simulation_and_cost(beta_m, v1, v2):
    """
    1) Computes v3 from v1 and v2.
    2) Solves the ODE.
    3) Returns the final cumulative cost and computed v3.
    """
    v3 = compute_v3(v1, v2, N, daily_vacc_capacity)
    if v3 < 0:
        return None
    # Adjust transmission based on beta_m
    beta_mod = beta_base * beta_m

    # Solve ODE
    results = odeint(deriv_sirv, initial_conditions, t, args=(N, beta_mod, gamma, W, v1, v2, v3))
    cost_df = compute_cost_dataframe(results, beta_m, N, dt=1.0)
    final_cost = cost_df["Cumulative_Cost"].iloc[-1]
    return (final_cost, v3)

# ----------------------------------------------------------------
# 4. Optimization: Find Best (v1, v2) for Each beta_m and Print Summary
# ----------------------------------------------------------------
beta_m_values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
v1_candidates = np.linspace(0.0, 0.04, 21)  # Candidate daily vaccination rate for children
v2_candidates = np.linspace(0.0, 0.02, 21)  # Candidate daily vaccination rate for adults

results_records = []

for bm in beta_m_values:
    best_cost_bm = float('inf')
    best_tuple_bm = None

    for v1 in v1_candidates:
        for v2 in v2_candidates:
            out = run_simulation_and_cost(bm, v1, v2)
            if out is None:
                continue
            final_cost, v3_val = out
            if final_cost < best_cost_bm:
                best_cost_bm = final_cost
                best_tuple_bm = (v1, v2, v3_val)

    if best_tuple_bm is None:
        print(f"No valid scenario found for beta_m = {bm}")
    else:
        v1_opt, v2_opt, v3_opt = best_tuple_bm
        results_records.append((bm, v1_opt, v2_opt, v3_opt, best_cost_bm))

# Print summary table
print("Best results for each beta_m:")
print("beta_m |   v1   |   v2   |   v3   |  Min Cost")
for row in results_records:
    bm, v1_o, v2_o, v3_o, cost_o = row
    print(f"{bm:5.2f}  | {v1_o:6.4f} | {v2_o:6.4f} | {v3_o:6.4f} | {cost_o:,.2f}")

# Save summary table to CSV
best_df = pd.DataFrame(results_records, columns=["beta_m", "v1_opt", "v2_opt", "v3_opt", "MinCost"])
best_df.to_csv("Figures/BestScenarioPerBetaM.csv", index=False)
print("\nSaved best scenarios to 'Figures/BestScenarioPerBetaM.csv'.")

# ----------------------------------------------------------------
# 5. Heat Maps: Build 2D Cost Arrays and Plot for Each beta_m
# ----------------------------------------------------------------
# Wrapper: run simulation and return only the final cost.
def run_simulation_age_alloc_fixedBM(beta_m, v1, v2):
    out = run_simulation_and_cost(beta_m, v1, v2)
    if out is None:
        return None
    final_cost, _ = out
    return final_cost

def build_cost_array_for_fixedBM(beta_m, v1_candidates, v2_candidates):
    """
    Returns a 2D numpy array Z with shape (len(v1_candidates), len(v2_candidates))
    where Z[i,j] is the final cumulative cost (or np.nan if invalid).
    """
    n1 = len(v1_candidates)
    n2 = len(v2_candidates)
    Z = np.full((n1, n2), np.nan)
    for i, v1 in enumerate(v1_candidates):
        for j, v2 in enumerate(v2_candidates):
            cost_val = run_simulation_age_alloc_fixedBM(beta_m, v1, v2)
            if cost_val is not None:
                Z[i, j] = cost_val
    return Z

# Create heat maps for each beta_m in a 3x3 grid.
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12), sharex=True, sharey=True)
fig.suptitle("Heat Maps of Final Cost vs (v1, v2) for Different beta_m", y=1.02)

pcm = None  # To hold the last pcolormesh for the colorbar

for idx, bm in enumerate(beta_m_values):
    ax = axes[idx // 3, idx % 3]
    # Build cost array for this beta_m
    Z = build_cost_array_for_fixedBM(bm, v1_candidates, v2_candidates)
    # Create meshgrid for plotting (v2 on x-axis, v1 on y-axis)
    X, Y = np.meshgrid(v2_candidates, v1_candidates, indexing="xy")
    
    # Use a colormap that shows NaNs as light gray
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgray")
    
    pc = ax.pcolormesh(X, Y, Z, cmap=cmap, shading='auto')
    ax.set_title(f"beta_m = {bm:.1f}")
    ax.set_xlabel("v2 (Adults)")
    ax.set_ylabel("v1 (Children)")
    
    pcm = pc

fig.tight_layout()
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(pcm, cax=cbar_ax, label="Total Cost (USD)")

plt.savefig("Figures/MultiHeatMaps_betaM.png", dpi=300, bbox_inches="tight")
plt.show()

# -*- coding: utf-8 -*-
"""
Combined Example: Optimization Printout & Profiles vs m
-------------------------------------------------------
This script runs a 3-group SIRV model simulation and computes day‐by‐day costs.
It then:

  1) Loops over (beta_m, v1, v2) to determine the best (minimum cost)
     vaccination scenario per beta_m, prints a summary table, and saves the
     results to CSV.
  2) Uses only the optimal (v1, v2, v3) for each beta_m to create four plots:
       - Optimal total cost vs m
       - v1*(m), v2*(m), v3*(m) vs m
     All four plots are vertically aligned, and the global minimum cost point
     is highlighted with a red vertical line and red markers.
  3) Prints which m gives the global minimum cost and how long the computation
     took.

Requirements: numpy, pandas, matplotlib, scipy.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# ----------------------------------------------------------------
# 0. (Optional) Create output folder if it does not exist
# ----------------------------------------------------------------
if not os.path.exists("Figures"):
    os.makedirs("Figures")

# ----------------------------------------------------------------
# 1. Population & Disease Model Parameters
# ----------------------------------------------------------------
# Age groups: children (0–19), adults (20–49), seniors (50+)
N = np.array([6246073, 31530507, 13972160])  # [N_children, N_adults, N_seniors]

# Initial conditions: all susceptible except one infected in each group
S_init = N.copy()
I_init = np.array([1, 1, 1])
R_init = np.array([0, 0, 0])
V_init = np.array([0, 0, 0])

# Initial state vector (length 12): [S1,I1,R1,V1, S2,I2,R2,V2, S3,I3,R3,V3]
initial_conditions = np.array([
    S_init[0], I_init[0], R_init[0], V_init[0],
    S_init[1], I_init[1], R_init[1], V_init[1],
    S_init[2], I_init[2], R_init[2], V_init[2]
])

# Recovery rates for children, adults, seniors
gamma = np.array([0.125, 0.083, 0.048])

# Waning immunity rate (applies to both R and V)
W = 0.005

# Baseline transmission matrix
beta_base = np.array([
    [0.1901, 0.0660, 0.0028],
    [0.1084, 0.1550, 0.0453],
    [0.1356, 0.1540, 0.1165]
])

# ----------------------------------------------------------------
# 2. Vaccination Capacity and Time Settings
# ----------------------------------------------------------------
total_pop = np.sum(N)

# Daily vaccination capacity (fraction of entire population per day)
daily_vacc_capacity = 0.01 * total_pop  # 1% of total population per day

# Time horizon for ODE solve (e.g., 150 days)
time_span = 150
t = np.linspace(0, time_span, time_span)  # integer days 0,1,...,149

# ----------------------------------------------------------------
# 3. Economic Parameters
# ----------------------------------------------------------------
# Medical costs per case (mild, hospitalized, ICU)
cost_mild = 160.8
cost_hosp = 432.5
cost_icu = 1129.5

# Fractions of mild, hospitalized, and ICU cases per group
f_child_mild, f_child_hosp, f_child_icu = 0.95, 0.04, 0.01
f_adult_mild, f_adult_hosp, f_adult_icu = 0.90, 0.08, 0.02
f_senior_mild, f_senior_hosp, f_senior_icu = 0.70, 0.20, 0.10

# Wage, funeral, and mortality parameters
employment_rates = np.array([0.02, 0.694, 0.513])
daily_income = np.array([27.74, 105.12, 92.45])
funeral_cost = 9405.38
mortality_fraction = np.array([0.000001, 0.00003, 0.007])

# Vaccination cost per person
vaccination_cost_per_capita = 35.76

# GDP loss model parameters
GDP_per_capita = 31929
a, b, c = -0.3648, -8.7989, -0.0012  # GDP loss fraction = a*exp(b*beta_m)+c

# ----------------------------------------------------------------
# 4. ODE System and Helper Functions
# ----------------------------------------------------------------
def deriv_sirv(y, t, N, beta_mod, gamma, W, v1, v2, v3):
    """
    SIRV ODE system for 3 age groups.
    y: 12-element vector [S1,I1,R1,V1, S2,I2,R2,V2, S3,I3,R3,V3]
    """
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y

    # Forces of infection for each group
    lambda1 = beta_mod[0, 0] * I1 / N[0] + beta_mod[0, 1] * I2 / N[1] + beta_mod[0, 2] * I3 / N[2]
    lambda2 = beta_mod[1, 0] * I1 / N[0] + beta_mod[1, 1] * I2 / N[1] + beta_mod[1, 2] * I3 / N[2]
    lambda3 = beta_mod[2, 0] * I1 / N[0] + beta_mod[2, 1] * I2 / N[1] + beta_mod[2, 2] * I3 / N[2]

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

    return [
        dS1dt, dI1dt, dR1dt, dV1dt,
        dS2dt, dI2dt, dR2dt, dV2dt,
        dS3dt, dI3dt, dR3dt, dV3dt
    ]


def compute_v3(v1, v2, N, daily_vacc_capacity):
    """
    Given v1 and v2 (children and adults), compute v3 for seniors so that
    N1*v1 + N2*v2 + N3*v3 = daily_vacc_capacity.

    Returns the implied v3 (can be negative if (v1, v2) exceed capacity).
    """
    numer = daily_vacc_capacity - (N[0] * v1 + N[1] * v2)
    return numer / N[2]


def compute_cost_dataframe(results, beta_m, N, dt=1.0):
    """
    Given the ODE solution 'results' and beta_m, compute:
      - daily medical, wage, death, vaccination, and GDP loss costs
      - daily total cost
      - cumulative sums of all cost components

    Returns a pandas DataFrame with time, S/I/R/V for each group,
    and all cost columns.
    """
    T = results.shape[0]
    times = np.arange(T) * dt

    med_arr = np.zeros(T)
    wage_arr = np.zeros(T)
    death_arr = np.zeros(T)
    vacc_arr = np.zeros(T)
    gdp_arr = np.zeros(T)
    total_arr = np.zeros(T)

    cum_med_arr = np.zeros(T)
    cum_wage_arr = np.zeros(T)
    cum_death_arr = np.zeros(T)
    cum_vacc_arr = np.zeros(T)
    cum_gdp_arr = np.zeros(T)
    cum_total_arr = np.zeros(T)

    # Daily GDP loss (constant per day for fixed beta_m)
    GDP_loss_fraction = a * np.exp(b * beta_m) + c
    GDP_loss_rate = GDP_per_capita * abs(GDP_loss_fraction) * (np.sum(N) / 365.0)

    for i in range(T):
        S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = results[i]

        # Infected counts by group
        I_child, I_adult, I_senior = I1, I2, I3

        # Medical costs
        med_child = I_child * (f_child_mild * cost_mild +
                               f_child_hosp * cost_hosp +
                               f_child_icu * cost_icu)
        med_adult = I_adult * (f_adult_mild * cost_mild +
                               f_adult_hosp * cost_hosp +
                               f_adult_icu * cost_icu)
        med_senior = I_senior * (f_senior_mild * cost_mild +
                                 f_senior_hosp * cost_hosp +
                                 f_senior_icu * cost_icu)
        med = med_child + med_adult + med_senior

        # Wage loss costs
        w_child = daily_income[0] * employment_rates[0] * I_child
        w_care = daily_income[1] * employment_rates[1] * I_child  # caregiver for sick children
        w_adult = daily_income[1] * employment_rates[1] * I_adult
        w_senior = daily_income[2] * employment_rates[2] * I_senior
        wage = w_child + w_care + w_adult + w_senior

        # Funeral (death) costs
        death = funeral_cost * (
            mortality_fraction[0] * I_child +
            mortality_fraction[1] * I_adult +
            mortality_fraction[2] * I_senior
        )

        # Vaccination costs (proportional to current number vaccinated)
        vacc = vaccination_cost_per_capita * (V1 + V2 + V3)

        # Daily GDP loss
        gdp_daily = GDP_loss_rate

        # Total daily cost
        total = med + wage + death + vacc + gdp_daily

        # Store daily values
        med_arr[i] = med
        wage_arr[i] = wage
        death_arr[i] = death
        vacc_arr[i] = vacc
        gdp_arr[i] = gdp_daily
        total_arr[i] = total

        # Update cumulative sums
        if i == 0:
            cum_med_arr[i] = med
            cum_wage_arr[i] = wage
            cum_death_arr[i] = death
            cum_vacc_arr[i] = vacc
            cum_gdp_arr[i] = gdp_daily
            cum_total_arr[i] = total
        else:
            cum_med_arr[i] = cum_med_arr[i - 1] + med
            cum_wage_arr[i] = cum_wage_arr[i - 1] + wage
            cum_death_arr[i] = cum_death_arr[i - 1] + death
            cum_vacc_arr[i] = cum_vacc_arr[i - 1] + vacc
            cum_gdp_arr[i] = cum_gdp_arr[i - 1] + gdp_daily
            cum_total_arr[i] = cum_total_arr[i - 1] + total

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
    For given (beta_m, v1, v2):

      1) Compute v3 from the vaccine capacity constraint.
      2) Solve the SIRV ODE system.
      3) Compute the cost DataFrame and return the final cumulative cost and v3.

    Returns:
      (final_cumulative_cost, v3) if v3 >= 0,
      None if v3 < 0 (invalid allocation exceeding capacity).
    """
    v3 = compute_v3(v1, v2, N, daily_vacc_capacity)

    # If v3 is negative, total vaccine demand exceeds capacity: mark as invalid
    if v3 < 0:
        return None

    # Modify transmission matrix according to beta_m
    beta_mod = beta_base * beta_m

    # Solve ODE
    results = odeint(
        deriv_sirv, initial_conditions, t,
        args=(N, beta_mod, gamma, W, v1, v2, v3)
    )

    # Compute costs
    cost_df = compute_cost_dataframe(results, beta_m, N, dt=1.0)
    final_cost = cost_df["Cumulative_Cost"].iloc[-1]

    return final_cost, v3

# ----------------------------------------------------------------
# 5. Optimization: Find Best (v1, v2, v3) for Each beta_m
# ----------------------------------------------------------------
# Start timer for optimization + plotting
start_time = time.time()

# m values
# beta_m_values = np.arange(0.2, 1.01, 0.1)
beta_m_values = np.arange(0.2, 1.01, 0.01)


# Candidate daily vaccination rates for children and adults
v1_candidates = np.linspace(0.0, 0.04, 21)  # children
v2_candidates = np.linspace(0.0, 0.02, 21)  # adults

# v1_candidates = np.linspace(0.0, 0.04, 81)  # children
# v2_candidates = np.linspace(0.0, 0.02, 81)  # adults

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
        print(f"No valid scenario found for beta_m = {bm:.2f}")
    else:
        v1_opt, v2_opt, v3_opt = best_tuple_bm
        results_records.append((bm, v1_opt, v2_opt, v3_opt, best_cost_bm))

# Print summary table of best scenarios
print("Best results for each beta_m:")
print("beta_m |   v1   |   v2   |   v3   |      Min Cost")
for row in results_records:
    bm, v1_o, v2_o, v3_o, cost_o = row
    print(f"{bm:5.2f}  | {v1_o:6.4f} | {v2_o:6.4f} | {v3_o:6.4f} | {cost_o:,.2f}")

# Save summary table to CSV
best_df = pd.DataFrame(
    results_records,
    columns=["beta_m", "v1_opt", "v2_opt", "v3_opt", "MinCost"]
)
best_df.to_csv("Figures/BestScenarioPerBetaM.csv", index=False)
print("\nSaved best scenarios to 'Figures/BestScenarioPerBetaM.csv'.")

# ----------------------------------------------------------------
# 6. Profiles vs m: Optimal Cost, v1, v2, v3 as Functions of m
# ----------------------------------------------------------------
# Sort by beta_m to ensure increasing m on the x-axis
best_df_sorted = best_df.sort_values("beta_m")

m_vals = best_df_sorted["beta_m"].values
v1_opt = best_df_sorted["v1_opt"].values
v2_opt = best_df_sorted["v2_opt"].values
v3_opt = best_df_sorted["v3_opt"].values
min_cost = best_df_sorted["MinCost"].values

# Find global minimum cost across all m values
global_min_idx = np.argmin(min_cost)
m_star = m_vals[global_min_idx]
cost_star = min_cost[global_min_idx]
v1_star = v1_opt[global_min_idx]
v2_star = v2_opt[global_min_idx]
v3_star = v3_opt[global_min_idx]

# Print which m gives the global minimum cost and the corresponding rates
print("\nGlobal minimum over all m:")
print(f"  m*       = {m_star:.2f}")
print(f"  v1*(m*)  = {v1_star:.4f}")
print(f"  v2*(m*)  = {v2_star:.4f}")
print(f"  v3*(m*)  = {v3_star:.4f}")
print(f"  MinCost* = {cost_star:,.2f}")

# Convert cost to billions for plotting in the first (top) panel
min_cost_billion = min_cost / 1e9
cost_star_billion = cost_star / 1e9

# Create four vertically stacked plots:
# 1) Optimal total cost vs m (in billions of dollars)
# 2) v1*(m)
# 3) v2*(m)
# 4) v3*(m)
fig, axes = plt.subplots(
    nrows=4, ncols=1,
    figsize=(6, 10),
    sharex=True
)

# --- 1) Optimal total cost vs m (in billions) ---
ax0 = axes[0]
ax0.plot(m_vals, min_cost_billion, marker="o")
ax0.scatter(m_star, cost_star_billion, color="red", zorder=3)
ax0.set_ylabel("Optimal cost\n(billion dollars)")
ax0.set_title("Optimal cost and vaccination rates vs m")

# --- 2) v1(m) ---
ax1 = axes[1]
ax1.plot(m_vals, v1_opt, marker="o")
ax1.scatter(m_star, v1_star, color="red", zorder=3)
ax1.set_ylabel("v1 (children)")

# --- 3) v2(m) ---
ax2 = axes[2]
ax2.plot(m_vals, v2_opt, marker="o")
ax2.scatter(m_star, v2_star, color="red", zorder=3)
ax2.set_ylabel("v2 (adults)")

# --- 4) v3(m) ---
ax3 = axes[3]
ax3.plot(m_vals, v3_opt, marker="o")
ax3.scatter(m_star, v3_star, color="red", zorder=3)
ax3.set_ylabel("v3 (seniors)")
ax3.set_xlabel("m (beta multiplier)")

# Draw a red vertical dashed line at the global minimum m across all subplots
for ax in axes:
    ax.axvline(m_star, color="red", linestyle="--", alpha=0.7)

fig.tight_layout()
plt.savefig("Figures/OptimalProfiles_vs_m.png", dpi=300, bbox_inches="tight")
plt.show()

# ----------------------------------------------------------------
# 7. Print total computation time
# ----------------------------------------------------------------
elapsed_time = time.time() - start_time
print(f"\nTotal computation time (optimization + plotting): "
      f"{elapsed_time:.2f} seconds "
      f"({elapsed_time/60.0:.2f} minutes).")

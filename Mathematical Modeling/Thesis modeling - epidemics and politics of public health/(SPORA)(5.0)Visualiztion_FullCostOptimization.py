# -*- coding: utf-8 -*-
"""
(SPORA)(4.0) SIRVD Cost Optimization and Visualization
------------------------------------------------------
This script runs a 3-group age-structured SIRVD model and computes
daily and cumulative economic costs. It then:

  1) Loops over (beta_m, v1, v2) to determine, for each transmission
     multiplier beta_m, the vaccination allocation (v1, v2, v3) that
     minimizes the total cumulative cost over the simulation window.
  2) Stores the optimal vaccination rates and minimal costs for each
     beta_m in a CSV file.
  3) Produces vertically stacked plots showing:
        - optimal total cost vs beta_m
        - optimal v1(beta_m), v2(beta_m), v3(beta_m)
     with the global minimum highlighted.
  4) Produces a 3×3 panel of heat maps of final total cost vs (v1, v2)
     for selected beta_m values, with the optimal (v1, v2) marked.

Key modeling change relative to the original SIRV version:
  • We now include an explicit death compartment D_i for each age group.
  • Infectious individuals in group i move to D_i at rate mu_i:
        dI_i/dt = ... - mu_i * I_i
        dD_i/dt =      mu_i * I_i
  • The death cost is computed from the *increment* in D_i between
    consecutive time steps, multiplied by the funeral cost FC.
    This ensures:
        C_death(0,T) = FC * sum_i D_i(T)
  • All other cost components (medical, wage loss, vaccination, GDP loss)
    follow the same definitions as in the original SIRV-based code.

Requirements: numpy, pandas, matplotlib, scipy.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --------------------------------------------------------------------
# 0. Output folder and figure-saving helper
# --------------------------------------------------------------------
# All figures and CSV summaries will be saved under this folder.
FIG_DIR = "(SPORA)Figures"
os.makedirs(FIG_DIR, exist_ok=True)

def save_figure(filename_base, dpi=300):
    """
    Save the current matplotlib figure to (SPORA)Figures with a given base name.

    Example
    -------
    save_figure("OptimalProfiles_vs_m_SIRVD")
    -> saves "(SPORA)Figures/OptimalProfiles_vs_m_SIRVD.png"
    """
    full_path = os.path.join(FIG_DIR, f"{filename_base}.png")
    plt.savefig(full_path, dpi=dpi, bbox_inches="tight")
    print(f"[Figure saved] {full_path}")


# --------------------------------------------------------------------
# 1. Population & Disease Model Parameters
# --------------------------------------------------------------------
# Age groups:
#   group 1: children (0–19)
#   group 2: adults   (20–49)
#   group 3: seniors  (50+)
N = np.array([6246073, 31530507, 13972160])  # [N_children, N_adults, N_seniors]

# Initial conditions:
#   – Start with everyone susceptible except one infected individual
#     in each age group.
#   – No one is recovered, vaccinated, or dead at t = 0.
S_init = N.copy()
I_init = np.array([1, 1, 1])
R_init = np.array([0, 0, 0])
V_init = np.array([0, 0, 0])
D_init = np.array([0, 0, 0])

# State vector y has length 15:
#   y = [S1,I1,R1,V1,D1, S2,I2,R2,V2,D2, S3,I3,R3,V3,D3]
initial_conditions = np.array([
    S_init[0], I_init[0], R_init[0], V_init[0], D_init[0],
    S_init[1], I_init[1], R_init[1], V_init[1], D_init[1],
    S_init[2], I_init[2], R_init[2], V_init[2], D_init[2]
])

# Recovery rates for children, adults, seniors (per day)
gamma = np.array([0.125, 0.083, 0.048])

# Waning immunity rate (applies to both R_i and V_i for all groups)
W = 0.005

# Disease-induced mortality rates mu_i (per day) for children, adults, seniors.
# In the ODE system, 1/mu_i is interpreted as the average number of days
# an infected individual in group i takes to die (conditional on eventual death).
mu = np.array([0.000001, 0.00003, 0.007])

# Baseline (pre-intervention) transmission matrix \hat{beta}
beta_base = np.array([
    [0.1901, 0.0660, 0.0028],
    [0.1084, 0.1550, 0.0453],
    [0.1356, 0.1540, 0.1165]
])

# --------------------------------------------------------------------
# 2. Vaccination Capacity and Time Settings
# --------------------------------------------------------------------
total_pop = np.sum(N)

# Daily vaccination capacity:
#   Here we assume we can vaccinate 1% of the entire population per day.
daily_vacc_capacity = 0.01 * total_pop  # 1% of total population per day

# ODE time horizon (e.g., 150 days)
time_span = 150
# Use one time unit per day; t = 0,1,...,149
t = np.linspace(0, time_span, time_span)

# --------------------------------------------------------------------
# 3. Economic Parameters
# --------------------------------------------------------------------
# Medical costs per case (mild, hospitalized, ICU) in USD per day
cost_mild = 160.8
cost_hosp = 432.5
cost_icu = 1129.5

# Fractions of mild, hospitalized, and ICU cases per age group
# These fractions sum to 1.0 for each row (age group) and determine
# how I_i(t) is partitioned into severity categories.
f_child_mild,  f_child_hosp,  f_child_icu  = 0.95, 0.04, 0.01
f_adult_mild,  f_adult_hosp,  f_adult_icu  = 0.90, 0.08, 0.02
f_senior_mild, f_senior_hosp, f_senior_icu = 0.70, 0.20, 0.10

# Daily employment rates and incomes for each age group
employment_rates = np.array([0.02, 0.694, 0.513])
daily_income     = np.array([27.74, 105.12, 92.45])

# Funeral cost per death (USD)
funeral_cost = 9405.38

# Vaccination cost per person (USD)
vaccination_cost_per_capita = 35.76

# GDP loss model parameters:
#   GDP loss fraction L(m) = a * exp(b * m) + c
#   These values come from a separate exponential fit using
#   three observed (m, GDP-loss) data points.
GDP_per_capita = 31929
a, b, c = -0.3648, -8.7989, -0.0012

# --------------------------------------------------------------------
# 4. SIRVD ODE System and Helper Functions
# --------------------------------------------------------------------
def deriv_sirvd(y, t, N, beta_mod, gamma, W, v1, v2, v3, mu):
    """
    SIRVD ODE system for 3 age groups with explicit death compartments.

    Parameters
    ----------
    y : array, shape (15,)
        Current state vector:
          [S1,I1,R1,V1,D1, S2,I2,R2,V2,D2, S3,I3,R3,V3,D3].
    t : float
        Current time (days).
    N : array, shape (3,)
        Population sizes for children, adults, seniors.
    beta_mod : array, shape (3,3)
        Modified transmission matrix, beta_mod = beta_m * beta_base.
    gamma : array, shape (3,)
        Recovery rates for each age group.
    W : float
        Waning immunity rate (applies to R_i and V_i).
    v1, v2, v3 : float
        Vaccination rates (per day) for each age group.
    mu : array, shape (3,)
        Disease-induced mortality rates for each age group.

    Returns
    -------
    dydt : list of length 15
        Time derivatives of the state variables.
    """
    S1, I1, R1, V1, D1, \
    S2, I2, R2, V2, D2, \
    S3, I3, R3, V3, D3 = y

    # Forces of infection (lambda_i) for each age group.
    # Note: we use N[i] (live population + deaths) in the denominator,
    #       consistent with the original SIRV calibration.
    lambda1 = beta_mod[0, 0] * I1 / N[0] + beta_mod[0, 1] * I2 / N[1] + beta_mod[0, 2] * I3 / N[2]
    lambda2 = beta_mod[1, 0] * I1 / N[0] + beta_mod[1, 1] * I2 / N[1] + beta_mod[1, 2] * I3 / N[2]
    lambda3 = beta_mod[2, 0] * I1 / N[0] + beta_mod[2, 1] * I2 / N[1] + beta_mod[2, 2] * I3 / N[2]

    # ---- Children (group 1) ----
    dS1dt = -lambda1 * S1 - v1 * S1 + W * (R1 + V1)
    dI1dt =  lambda1 * S1 - gamma[0] * I1 - mu[0] * I1
    dR1dt =  gamma[0] * I1 - W * R1
    dV1dt =  v1 * S1 - W * V1
    dD1dt =  mu[0] * I1  # cumulative deaths in group 1

    # ---- Adults (group 2) ----
    dS2dt = -lambda2 * S2 - v2 * S2 + W * (R2 + V2)
    dI2dt =  lambda2 * S2 - gamma[1] * I2 - mu[1] * I2
    dR2dt =  gamma[1] * I2 - W * R2
    dV2dt =  v2 * S2 - W * V2
    dD2dt =  mu[1] * I2  # cumulative deaths in group 2

    # ---- Seniors (group 3) ----
    dS3dt = -lambda3 * S3 - v3 * S3 + W * (R3 + V3)
    dI3dt =  lambda3 * S3 - gamma[2] * I3 - mu[2] * I3
    dR3dt =  gamma[2] * I3 - W * R3
    dV3dt =  v3 * S3 - W * V3
    dD3dt =  mu[2] * I3  # cumulative deaths in group 3

    return [
        dS1dt, dI1dt, dR1dt, dV1dt, dD1dt,
        dS2dt, dI2dt, dR2dt, dV2dt, dD2dt,
        dS3dt, dI3dt, dR3dt, dV3dt, dD3dt
    ]


def compute_v3(v1, v2, N, daily_vacc_capacity):
    """
    Given vaccination rates v1 and v2 (children and adults), compute v3
    (seniors) so that the total number of vaccinations per day satisfies
        N1*v1 + N2*v2 + N3*v3 = daily_vacc_capacity.

    If the implied v3 is negative, the pair (v1, v2) violates the
    capacity constraint and will be treated as invalid.
    """
    numer = daily_vacc_capacity - (N[0] * v1 + N[1] * v2)
    return numer / N[2]


def compute_cost_dataframe(results, beta_m, N, dt=1.0):
    """
    Given the ODE solution 'results' and the transmission multiplier beta_m,
    compute:

      – daily medical, wage, death, vaccination, and GDP-loss costs,
      – daily total cost,
      – cumulative sums of all cost components.

    The crucial feature is the death cost:
      • D1(t), D2(t), D3(t) are cumulative deaths in each group.
      • We define daily deaths at step i as:
            new_deaths(i) = [D1+D2+D3]_i - [D1+D2+D3]_{i-1}
      • Then daily death cost is:
            C_death(i) = funeral_cost * new_deaths(i)
      • Summing C_death(i) over i yields
            funeral_cost * [ D1(T) + D2(T) + D3(T) ].

    Returns
    -------
    df : pandas.DataFrame
        Contains time, S/I/R/V/D for each group, daily cost components,
        and cumulative cost components.
    """
    T = results.shape[0]
    times = np.arange(T) * dt

    # Arrays for daily costs
    med_arr   = np.zeros(T)
    wage_arr  = np.zeros(T)
    death_arr = np.zeros(T)
    vacc_arr  = np.zeros(T)
    gdp_arr   = np.zeros(T)
    total_arr = np.zeros(T)

    # Arrays for cumulative costs
    cum_med_arr   = np.zeros(T)
    cum_wage_arr  = np.zeros(T)
    cum_death_arr = np.zeros(T)
    cum_vacc_arr  = np.zeros(T)
    cum_gdp_arr   = np.zeros(T)
    cum_total_arr = np.zeros(T)

    # For a fixed beta_m, GDP loss per day is constant.
    GDP_loss_fraction = a * np.exp(b * beta_m) + c
    # Interpret |GDP_loss_fraction| as the annual GDP fraction lost,
    # converted to a daily amount:
    GDP_loss_rate = GDP_per_capita * abs(GDP_loss_fraction) * (np.sum(N) / 365.0)

    # Previous day's cumulative deaths, used to compute daily increments
    prev_D1 = prev_D2 = prev_D3 = 0.0

    for i in range(T):
        S1, I1, R1, V1, D1, \
        S2, I2, R2, V2, D2, \
        S3, I3, R3, V3, D3 = results[i]

        # ---- Medical cost ----
        I_child, I_adult, I_senior = I1, I2, I3

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

        # ---- Wage loss ----
        # Lost productivity from infected individuals, plus an additional
        # caregiver term for adults taking care of sick children.
        w_child  = daily_income[0] * employment_rates[0] * I_child
        w_care   = daily_income[1] * employment_rates[1] * I_child
        w_adult  = daily_income[1] * employment_rates[1] * I_adult
        w_senior = daily_income[2] * employment_rates[2] * I_senior
        wage = w_child + w_care + w_adult + w_senior

        # ---- Death cost (via D compartments) ----
        # Total cumulative deaths at time step i
        total_D = D1 + D2 + D3
        if i == 0:
            # On the first day, the increment equals the current total
            new_deaths = total_D
        else:
            prev_total_D = prev_D1 + prev_D2 + prev_D3
            new_deaths = max(total_D - prev_total_D, 0.0)
        death = funeral_cost * new_deaths

        # Update previous D values for the next step
        prev_D1, prev_D2, prev_D3 = D1, D2, D3

        # ---- Vaccination cost ----
        # Proportional to the current number of vaccinated individuals.
        vacc = vaccination_cost_per_capita * (V1 + V2 + V3)

        # ---- GDP loss ----
        gdp_daily = GDP_loss_rate

        # ---- Total daily cost ----
        total = med + wage + death + vacc + gdp_daily

        # Store daily costs
        med_arr[i]   = med
        wage_arr[i]  = wage
        death_arr[i] = death
        vacc_arr[i]  = vacc
        gdp_arr[i]   = gdp_daily
        total_arr[i] = total

        # Update cumulative costs
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

    # Collect everything into a DataFrame
    df = pd.DataFrame({
        "time": times,
        "S1": results[:, 0],
        "I1": results[:, 1],
        "R1": results[:, 2],
        "V1": results[:, 3],
        "D1": results[:, 4],
        "S2": results[:, 5],
        "I2": results[:, 6],
        "R2": results[:, 7],
        "V2": results[:, 8],
        "D2": results[:, 9],
        "S3": results[:, 10],
        "I3": results[:, 11],
        "R3": results[:, 12],
        "V3": results[:, 13],
        "D3": results[:, 14],
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
    For a given (beta_m, v1, v2):

      1) Compute v3 from the vaccination capacity constraint.
      2) Solve the SIRVD ODE system over the pre-defined time grid t.
      3) Compute the cost DataFrame and return:
           – the final cumulative cost at t = T,
           – the implied v3.

    If v3 < 0 (which violates the daily capacity constraint),
    return None to indicate an invalid scenario.
    """
    # Compute senior vaccination rate from capacity
    v3 = compute_v3(v1, v2, N, daily_vacc_capacity)

    if v3 < 0:
        # Invalid choice: vaccination demands exceed daily capacity
        return None

    # Scale baseline transmission matrix by beta_m
    beta_mod = beta_base * beta_m

    # Solve SIRVD ODEs
    results = odeint(
        deriv_sirvd,
        initial_conditions,
        t,
        args=(N, beta_mod, gamma, W, v1, v2, v3, mu)
    )

    # Compute economic costs over time
    cost_df = compute_cost_dataframe(results, beta_m, N, dt=1.0)
    final_cost = cost_df["Cumulative_Cost"].iloc[-1]

    return final_cost, v3


# --------------------------------------------------------------------
# 5. Optimization: Best (v1, v2, v3) for Each beta_m
# --------------------------------------------------------------------
start_time = time.time()

# Range of beta_m (transmission multipliers) to test.
# Here we use a fine grid: 0.20, 0.21, ..., 1.00
beta_m_values = np.arange(0.2, 1.01, 0.01)

# Candidate daily vaccination rates for children and adults
# (v3 is computed from the capacity constraint)
v1_candidates = np.linspace(0.0, 0.04, 21)  # children
v2_candidates = np.linspace(0.0, 0.02, 21)  # adults

# Records of optimal scenario for each beta_m:
#   (beta_m, v1_opt, v2_opt, v3_opt, MinCost)
results_records = []

for bm in beta_m_values:
    best_cost_bm = float('inf')
    best_tuple_bm = None

    for v1 in v1_candidates:
        for v2 in v2_candidates:
            out = run_simulation_and_cost(bm, v1, v2)
            if out is None:
                # Exceeds vaccine capacity; skip
                continue

            final_cost, v3_val = out

            if final_cost < best_cost_bm:
                best_cost_bm = final_cost
                best_tuple_bm = (v1, v2, v3_val)

    if best_tuple_bm is None:
        print(f"[Warning] No valid scenario found for beta_m = {bm:.2f}")
    else:
        v1_opt, v2_opt, v3_opt = best_tuple_bm
        results_records.append((bm, v1_opt, v2_opt, v3_opt, best_cost_bm))

# Convert results to DataFrame and print a summary table
best_df = pd.DataFrame(
    results_records,
    columns=["beta_m", "v1_opt", "v2_opt", "v3_opt", "MinCost"]
)

print("Best results for each beta_m (including explicit death compartment D):")
print("beta_m |   v1   |   v2   |   v3   |      Min Cost")
for row in results_records:
    bm, v1_o, v2_o, v3_o, cost_o = row
    print(f"{bm:5.2f}  | {v1_o:6.4f} | {v2_o:6.4f} | {v3_o:6.4f} | {cost_o:,.2f}")

# Save the summary table to CSV (inside (SPORA)Figures)
csv_path = os.path.join(FIG_DIR, "BestScenarioPerBetaM_SIRVD.csv")
best_df.to_csv(csv_path, index=False)
print(f"\n[CSV saved] {csv_path}")

# --------------------------------------------------------------------
# 6. Profiles vs beta_m: Optimal Cost and Vaccination Rates
# --------------------------------------------------------------------
# Sort by beta_m to ensure monotone x-axis
best_df_sorted = best_df.sort_values("beta_m")

m_vals   = best_df_sorted["beta_m"].values
v1_opt   = best_df_sorted["v1_opt"].values
v2_opt   = best_df_sorted["v2_opt"].values
v3_opt   = best_df_sorted["v3_opt"].values
min_cost = best_df_sorted["MinCost"].values

# Global minimum over all beta_m values
global_min_idx = np.argmin(min_cost)
m_star    = m_vals[global_min_idx]
cost_star = min_cost[global_min_idx]
v1_star   = v1_opt[global_min_idx]
v2_star   = v2_opt[global_min_idx]
v3_star   = v3_opt[global_min_idx]

print("\nGlobal minimum over all beta_m (with explicit D):")
print(f"  m*       = {m_star:.2f}")
print(f"  v1*(m*)  = {v1_star:.4f}")
print(f"  v2*(m*)  = {v2_star:.4f}")
print(f"  v3*(m*)  = {v3_star:.4f}")
print(f"  MinCost* = {cost_star:,.2f}")

# Convert cost to billions of dollars for the top panel
min_cost_billion  = min_cost / 1e9
cost_star_billion = cost_star / 1e9

# Four vertically stacked plots:
#   1) optimal total cost vs beta_m (in billions)
#   2) v1*(beta_m)
#   3) v2*(beta_m)
#   4) v3*(beta_m)
fig, axes = plt.subplots(
    nrows=4,
    ncols=1,
    figsize=(6, 10),
    sharex=True
)

# ---- 1) Optimal total cost vs beta_m (billions of USD) ----
ax0 = axes[0]
ax0.plot(m_vals, min_cost_billion, marker="o")
ax0.scatter(m_star, cost_star_billion, color="red", zorder=3)
ax0.set_ylabel("Optimal cost\n(billion USD)")
ax0.set_title("Optimal cost and vaccination rates vs m (SIRVD)")

# ---- 2) v1(beta_m) ----
ax1 = axes[1]
ax1.plot(m_vals, v1_opt, marker="o")
ax1.scatter(m_star, v1_star, color="red", zorder=3)
ax1.set_ylabel("v1 (children)")

# ---- 3) v2(beta_m) ----
ax2 = axes[2]
ax2.plot(m_vals, v2_opt, marker="o")
ax2.scatter(m_star, v2_star, color="red", zorder=3)
ax2.set_ylabel("v2 (adults)")

# ---- 4) v3(beta_m) ----
ax3 = axes[3]
ax3.plot(m_vals, v3_opt, marker="o")
ax3.scatter(m_star, v3_star, color="red", zorder=3)
ax3.set_ylabel("v3 (seniors)")
ax3.set_xlabel("m (transmission multiplier)")

# Draw a vertical dashed line at the global minimum m across all panels
for ax in axes:
    ax.axvline(m_star, color="red", linestyle="--", alpha=0.7)

fig.tight_layout()
save_figure("OptimalProfiles_vs_m_SIRVD")
plt.show()

# --------------------------------------------------------------------
# 7. Heat Maps: Total cost over (v1, v2) for several beta_m values
# --------------------------------------------------------------------
# This section re-uses `run_simulation_and_cost(m, v1, v2)`,
# which already includes the SIRVD model and the updated death cost
# based on the D_i(T) compartment.

# Helper: wrapper that returns only the final total cost
def run_simulation_age_alloc_fixed_m(m, v1, v2):
    """
    Run one simulation for fixed (m, v1, v2) and return
    the final cumulative total cost.

    Returns
    -------
    float or None
        Final cumulative cost if the allocation is feasible (v3 >= 0),
        otherwise None.
    """
    out = run_simulation_and_cost(m, v1, v2)
    if out is None:
        return None
    final_cost, _ = out  # we ignore v3 here
    return final_cost


def build_cost_array_for_fixed_m(m, v1_candidates, v2_candidates):
    """
    Build a 2D array Z such that

        Z[i, j] = final total cost for (v1 = v1_candidates[i],
                                        v2 = v2_candidates[j])

    If the implied v3 < 0 (capacity exceeded), the entry is set to NaN.
    """
    n1 = len(v1_candidates)
    n2 = len(v2_candidates)
    Z = np.full((n1, n2), np.nan)

    for i, v1 in enumerate(v1_candidates):
        for j, v2 in enumerate(v2_candidates):
            cost_val = run_simulation_age_alloc_fixed_m(m, v1, v2)
            if cost_val is not None:
                Z[i, j] = cost_val
    return Z


# Choose the m values and vaccination grids used for the heat maps
m_grid_values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

# Re-use the same vaccination grids as in the optimization
v1_candidates = np.linspace(0.0, 0.04, 21)  # children
v2_candidates = np.linspace(0.0, 0.02, 21)  # adults

# Pre-compute cost arrays for all m values
cost_arrays = []
for m_val in m_grid_values:
    Z = build_cost_array_for_fixed_m(m_val, v1_candidates, v2_candidates)
    cost_arrays.append(Z)

# Data ranges (for aspect-ratio correction)
v1_min, v1_max = v1_candidates[0], v1_candidates[-1]
v2_min, v2_max = v2_candidates[0], v2_candidates[-1]

# Scale factor so that the plotted domain (v1 range vs v2 range) looks square
# Here v1 in [0, 0.04], v2 in [0, 0.02], so scale ≈ 2.
scale = (v1_max - v1_min) / (v2_max - v2_min)

# --------------------------------------------------------------------
# 7.1 Plot the 3×3 panel of heat maps
# --------------------------------------------------------------------
fig, axes = plt.subplots(
    nrows=3, ncols=3,
    figsize=(11, 10),
    sharex=False, sharey=False
)

fig.suptitle(
    "Heat Maps of Final Total Cost vs (v1, v2)\nfor Different Transmission Multipliers m",
    y=1.03
)

# Use viridis; set NaNs to light gray (infeasible allocations)
cmap = plt.cm.viridis.copy()
cmap.set_bad(color="lightgray")

pc = None  # last pcolormesh, used to define the colorbar

for idx, (m_val, Z) in enumerate(zip(m_grid_values, cost_arrays)):
    ax = axes[idx // 3, idx % 3]

    # For plotting, scale v2 so that the panel is visually square:
    #   X_plot = v2 * scale, Y_plot = v1
    X_plot, Y_plot = np.meshgrid(v2_candidates * scale,
                                 v1_candidates,
                                 indexing="xy")

    pc = ax.pcolormesh(
        X_plot, Y_plot, Z,
        cmap=cmap,
        shading="auto"
    )

    # Mark the optimal (minimum-cost) point for this m, if it exists
    if not np.all(np.isnan(Z)):
        flat_idx = np.nanargmin(Z)
        i_min, j_min = np.unravel_index(flat_idx, Z.shape)
        v1_star_hm = v1_candidates[i_min]
        v2_star_hm = v2_candidates[j_min]

        ax.scatter(
            v2_star_hm * scale,  # scaled x
            v1_star_hm,          # y
            s=55,
            facecolors="none",
            edgecolors="white",
            linewidth=1.8,
            zorder=3
        )

    # Force each panel to be square in display coordinates
    ax.set_aspect("equal", adjustable="box")

    # x-ticks: in scaled coordinates, but labeled with original v2 values
    v2_ticks = np.linspace(v2_min, v2_max, 4)
    ax.set_xticks(v2_ticks * scale)
    ax.set_xticklabels([f"{v:.003f}" for v in v2_ticks])

    # y-ticks: directly in v1 units
    v1_ticks = np.linspace(v1_min, v1_max, 5)
    ax.set_yticks(v1_ticks)
    ax.set_yticklabels([f"{v:.003f}" for v in v1_ticks])

    ax.set_title(f"m = {m_val:.1f}")
    ax.set_xlabel("v2 (Adults)")
    ax.set_ylabel("v1 (Children)")

fig.tight_layout(rect=[0.0, 0.0, 0.90, 1.0])

# Colorbar on the right, showing total cost in USD
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.70])
fig.colorbar(pc, cax=cbar_ax, label="Total Cost (USD)")

save_figure("MultiHeatMaps_m_SIRVD_optimal_points")
plt.show()

# --------------------------------------------------------------------
# 8. Total computation time
# --------------------------------------------------------------------
elapsed_time = time.time() - start_time
print(f"\nTotal computation time (optimization + plotting): "
      f"{elapsed_time:.2f} seconds "
      f"({elapsed_time/60.0:.2f} minutes).")

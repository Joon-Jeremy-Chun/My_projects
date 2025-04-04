# -*- coding: utf-8 -*-
"""
Extended Example: SIRV Model with Separate Vaccination Rates (v1, v2, v3)
Subject to a daily vaccination capacity. We do a grid search over (v1, v2),
collect final costs, and produce multiple plots, including:
 - 3D Scatter
 - 3D Triangulated Surface
 - 3D Structured Surface
 - 2D Heat Map (pcolormesh)
ADAPT AS NEEDED for your environment or specific needs.

Now includes code for saving each figure to the "Figures" folder,
with filenames that contain the time_span and beta_m_test values.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from scipy.integrate import odeint
import matplotlib.colors as mcolors

# ----------------------------------------------------------------
# 1. CREATE OUTPUT FOLDER
# ----------------------------------------------------------------
if not os.path.exists("Figures"):
    os.makedirs("Figures")

# ----------------------------------------------------------------
# 2. DEFINE MODEL PARAMETERS
# ----------------------------------------------------------------
time_span = 150
t = np.linspace(0, time_span, time_span)  # days 0..179

beta_m_test   = 1.0   # or whichever multiplier you want to analyze

# We can build a short identifier string for file-saving
filename_suffix = f"{time_span}days_beta{beta_m_test}"

# Population by age group (Children=0–19, Adults=20–49, Seniors=50+)
N = np.array([6246073, 31530507, 13972160])

# Initial conditions
S_init = N.copy()
I_init = np.array([1, 1, 1])
R_init = np.array([0, 0, 0])
V_init = np.array([0, 0, 0])

initial_conditions = np.array([
    S_init[0], I_init[0], R_init[0], V_init[0],
    S_init[1], I_init[1], R_init[1], V_init[1],
    S_init[2], I_init[2], R_init[2], V_init[2]
])

gamma = np.array([0.125, 0.083, 0.048])  # recovery rates
W = 0.005                                # waning immunity

# Daily total vaccination capacity ~1.07% of entire population
total_pop = np.sum(N)
daily_vacc_capacity = 0.0107 * total_pop

beta = np.array([
    [0.1901, 0.0660, 0.0028],
    [0.1084, 0.1550, 0.0453],
    [0.1356, 0.1540, 0.1165]
])

# ----------------------------------------------------------------
# ECONOMIC PARAMETERS
# ----------------------------------------------------------------
cost_mild   = 160.8
cost_hosp   = 432.5
cost_icu    = 1129.5

f_child_mild  = 0.95
f_child_hosp  = 0.04
f_child_icu   = 0.01

f_adult_mild  = 0.90
f_adult_hosp  = 0.08
f_adult_icu   = 0.02

f_senior_mild = 0.70
f_senior_hosp = 0.20
f_senior_icu  = 0.10

employment_rates = np.array([0.02, 0.694, 0.513])
daily_income     = np.array([27.74, 105.12, 92.45])
funeral_cost     = 9405.38
mortality_fraction = np.array([0.000001, 0.00003, 0.007])
vaccination_cost_per_capita = 35.76
GDP_per_capita = 31929

# GDP loss model: fraction = a * exp(b*beta_m) + c
a, b, c = -0.3648, -8.7989, -0.0012

# ----------------------------------------------------------------
# 3. ODE FOR AGE-SPECIFIC VACCINATION
# ----------------------------------------------------------------
def deriv_age_alloc(y, t, N, beta_mod, gamma, W, v1, v2, v3):
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y

    lambda1 = beta_mod[0,0]*I1/N[0] + beta_mod[0,1]*I2/N[1] + beta_mod[0,2]*I3/N[2]
    lambda2 = beta_mod[1,0]*I1/N[0] + beta_mod[1,1]*I2/N[1] + beta_mod[1,2]*I3/N[2]
    lambda3 = beta_mod[2,0]*I1/N[0] + beta_mod[2,1]*I2/N[1] + beta_mod[2,2]*I3/N[2]

    dS1dt = -lambda1*S1 - v1*S1 + W*(R1 + V1)
    dI1dt = lambda1*S1 - gamma[0]*I1
    dR1dt = gamma[0]*I1 - W*R1
    dV1dt = v1*S1 - W*V1

    dS2dt = -lambda2*S2 - v2*S2 + W*(R2 + V2)
    dI2dt = lambda2*S2 - gamma[1]*I2
    dR2dt = gamma[1]*I2 - W*R2
    dV2dt = v2*S2 - W*V2

    dS3dt = -lambda3*S3 - v3*S3 + W*(R3 + V3)
    dI3dt = lambda3*S3 - gamma[2]*I3
    dR3dt = gamma[2]*I3 - W*R3
    dV3dt = v3*S3 - W*V3

    return [dS1dt, dI1dt, dR1dt, dV1dt,
            dS2dt, dI2dt, dR2dt, dV2dt,
            dS3dt, dI3dt, dR3dt, dV3dt]

# ----------------------------------------------------------------
# 4. COST COMPUTATION
# ----------------------------------------------------------------
def compute_cost_dataframe(results, beta_m, N, dt=1.0):
    T = len(results)
    times = np.arange(T)*dt

    med_cost_arr   = np.zeros(T)
    wage_loss_arr  = np.zeros(T)
    death_cost_arr = np.zeros(T)
    vacc_cost_arr  = np.zeros(T)
    gdp_loss_arr   = np.zeros(T)
    total_cost_arr = np.zeros(T)

    cum_med_arr   = np.zeros(T)
    cum_wage_arr  = np.zeros(T)
    cum_death_arr = np.zeros(T)
    cum_vacc_arr  = np.zeros(T)
    cum_gdp_arr   = np.zeros(T)
    cum_total_arr = np.zeros(T)

    GDP_loss_fraction = a * np.exp(b * beta_m) + c
    GDP_loss_rate = GDP_per_capita * abs(GDP_loss_fraction) * (np.sum(N)/365.0)

    for i in range(T):
        S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = results[i]

        # Infecteds
        I_child  = I1
        I_adult  = I2
        I_senior = I3

        # 1) Medical costs
        med_child = I_child*(f_child_mild*cost_mild + f_child_hosp*cost_hosp + f_child_icu*cost_icu)
        med_adult = I_adult*(f_adult_mild*cost_mild + f_adult_hosp*cost_hosp + f_adult_icu*cost_icu)
        med_senior= I_senior*(f_senior_mild*cost_mild+f_senior_hosp*cost_hosp+f_senior_icu*cost_icu)
        med = med_child + med_adult + med_senior

        # 2) Wage loss
        W_child  = daily_income[0] * employment_rates[0]
        W_adult  = daily_income[1] * employment_rates[1]
        W_senior = daily_income[2] * employment_rates[2]

        wage_child_loss     = W_child  * I_child
        wage_caregiver_loss = W_adult  * I_child
        wage_adult_loss     = W_adult  * I_adult
        wage_senior_loss    = W_senior * I_senior

        wage = wage_child_loss + wage_caregiver_loss + wage_adult_loss + wage_senior_loss

        # 3) Death cost
        death = funeral_cost * (
            mortality_fraction[0]*I_child +
            mortality_fraction[1]*I_adult +
            mortality_fraction[2]*I_senior
        )

        # 4) Vaccination cost
        vacc = vaccination_cost_per_capita*(V1 + V2 + V3)

        # 5) GDP loss
        gdp_daily = GDP_loss_rate
        total_rate = med + wage + death + vacc + gdp_daily

        # record daily
        med_cost_arr[i]   = med
        wage_loss_arr[i]  = wage
        death_cost_arr[i] = death
        vacc_cost_arr[i]  = vacc
        gdp_loss_arr[i]   = gdp_daily
        total_cost_arr[i] = total_rate

        # accumulate
        if i==0:
            cum_med_arr[i]   = med
            cum_wage_arr[i]  = wage
            cum_death_arr[i] = death
            cum_vacc_arr[i]  = vacc
            cum_gdp_arr[i]   = gdp_daily
            cum_total_arr[i] = total_rate
        else:
            cum_med_arr[i]   = cum_med_arr[i-1] + med
            cum_wage_arr[i]  = cum_wage_arr[i-1] + wage
            cum_death_arr[i] = cum_death_arr[i-1]+ death
            cum_vacc_arr[i]  = cum_vacc_arr[i-1] + vacc
            cum_gdp_arr[i]   = cum_gdp_arr[i-1] + gdp_daily
            cum_total_arr[i] = cum_total_arr[i-1] + total_rate

    df = pd.DataFrame({
        "time": times,
        "S1": results[:,0],
        "I1": results[:,1],
        "R1": results[:,2],
        "V1": results[:,3],
        "S2": results[:,4],
        "I2": results[:,5],
        "R2": results[:,6],
        "V2": results[:,7],
        "S3": results[:,8],
        "I3": results[:,9],
        "R3": results[:,10],
        "V3": results[:,11],
        "Med_Cost": med_cost_arr,
        "Wage_Loss": wage_loss_arr,
        "Death_Cost": death_cost_arr,
        "Vacc_Cost": vacc_cost_arr,
        "GDP_Loss": gdp_loss_arr,
        "Total_Rate": total_cost_arr,
        "Cumulative_Med_Cost": cum_med_arr,
        "Cumulative_Wage_Loss": cum_wage_arr,
        "Cumulative_Death_Cost": cum_death_arr,
        "Cumulative_Vacc_Cost": cum_vacc_arr,
        "Cumulative_GDP_Loss": cum_gdp_arr,
        "Cumulative_Cost": cum_total_arr
    })
    return df

# ----------------------------------------------------------------
# 5. HELPER: COMPUTE V3
# ----------------------------------------------------------------
def compute_v3(v1, v2, N, daily_vacc_capacity):
    numer = daily_vacc_capacity - (N[0]*v1 + N[1]*v2)
    return numer / N[2]

# ----------------------------------------------------------------
# 6. RUN SIMULATION
# ----------------------------------------------------------------
def run_simulation_age_alloc(v1, v2, beta_m=1.0):
    v3 = compute_v3(v1, v2, N, daily_vacc_capacity)
    if v3 < 0:
        return None
    beta_mod = beta * beta_m
    results = odeint(deriv_age_alloc, initial_conditions, t, args=(N, beta_mod, gamma, W, v1, v2, v3))
    cost_df = compute_cost_dataframe(results, beta_m=beta_m, N=N, dt=1.0)
    return cost_df

# ----------------------------------------------------------------
# 7. GRID SEARCH OVER (v1, v2)
# ----------------------------------------------------------------
v1_candidates = np.linspace(0.0, 0.04, 21)
v2_candidates = np.linspace(0.0, 0.02, 21)

all_records = []
best_cost = float('inf')
best_tuple = None

for v1 in v1_candidates:
    for v2 in v2_candidates:
        df_out = run_simulation_age_alloc(v1, v2, beta_m_test)
        if df_out is None:
            continue
        final_cost = df_out["Cumulative_Cost"].iloc[-1]
        all_records.append((v1, v2, final_cost))
        if final_cost < best_cost:
            best_cost = final_cost
            best_tuple = (v1, v2, df_out)

cost_df_3d = pd.DataFrame(all_records, columns=["v1", "v2", "cost"])
print(cost_df_3d.head())

if best_tuple is not None:
    v1_opt, v2_opt, df_opt = best_tuple
    v3_opt = compute_v3(v1_opt, v2_opt, N, daily_vacc_capacity)
    print(f"Optimal scenario: v1={v1_opt:.4f}, v2={v2_opt:.4f}, v3={v3_opt:.4f}")
    print(f"Minimum total cost = {best_cost:,.2f}")
else:
    print("No valid scenario found in the grid.")

# ----------------------------------------------------------------
# 8. PLOTTING
# ----------------------------------------------------------------

# 8A. 3D SCATTER
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(
    cost_df_3d["v1"],
    cost_df_3d["v2"],
    cost_df_3d["cost"],
    c=cost_df_3d["cost"], 
    cmap="viridis"
)
ax.set_xlabel("v1 (Children)")
ax.set_ylabel("v2 (Adults)")
ax.set_zlabel("Total Cost")
plt.title("3D Scatter: Cost vs (v1, v2)")
cb = plt.colorbar(sc, pad=0.1)
cb.set_label("Total Cost")

# SAVE FIGURE
plt.savefig(f"Figures/3D_Scatter_{filename_suffix}.png", dpi=300, bbox_inches="tight")
plt.show()


# 8B. 3D TRIANGULATED SURFACE
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(
    cost_df_3d["v1"], 
    cost_df_3d["v2"], 
    cost_df_3d["cost"],
    cmap="viridis",
    edgecolor="none"
)
ax.set_xlabel("v1 (Children)")
ax.set_ylabel("v2 (Adults)")
ax.set_zlabel("Total Cost")
plt.title("3D TriSurf: Cost vs (v1, v2)")

# SAVE FIGURE
plt.savefig(f"Figures/3D_TriSurf_{filename_suffix}.png", dpi=300, bbox_inches="tight")
plt.show()


# 8C. 3D STRUCTURED SURFACE (NaN)
grid_size = len(v1_candidates)
Z = np.full((grid_size, grid_size), np.nan)

v1_to_index = {v: i for i, v in enumerate(v1_candidates)}
v2_to_index = {v: j for j, v in enumerate(v2_candidates)}

for (v1_val, v2_val, cost_val) in all_records:
    i = v1_to_index[v1_val]
    j = v2_to_index[v2_val]
    Z[i,j] = cost_val

X, Y = np.meshgrid(v2_candidates, v1_candidates, indexing="xy")

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor='none')
ax.set_xlabel("v2 (Adults)")
ax.set_ylabel("v1 (Children)")
ax.set_zlabel("Total Cost")
plt.title("3D Structured Surface (NaN for invalid)")

plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# SAVE FIGURE
plt.savefig(f"Figures/3D_Structured_Surface_{filename_suffix}.png", dpi=300, bbox_inches="tight")
plt.show()


# 8D. 2D HEAT MAP (pcolormesh)
cmap = plt.cm.get_cmap("viridis").copy()
cmap.set_bad(color="lightgray")

fig, ax = plt.subplots(figsize=(8,6))
pc = ax.pcolormesh(X, Y, Z, cmap=cmap, shading="auto")
ax.set_xlabel("v2 (Adults)")
ax.set_ylabel("v1 (Children)")
plt.title("Heat Map: Total Cost vs (v1, v2)")

cbar = plt.colorbar(pc, ax=ax)
cbar.set_label("Total Cost")

# SAVE FIGURE
plt.savefig(f"Figures/Heatmap_{filename_suffix}.png", dpi=300, bbox_inches="tight")
plt.show()

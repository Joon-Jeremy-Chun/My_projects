# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 23:44:36 2025
@author: joonc

Day-by-day summation of cost components in an SIRV model with piecewise vaccination,
now updated to incorporate separate costs for mild, hospitalized, and ICU cases,
with age-group-specific severity fractions (e.g., 0–19, 20–49, 50+).
"""

import os
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# -----------------------------
# Create output folder if needed
# -----------------------------
if not os.path.exists("Figures"):
    os.makedirs("Figures")

# -----------------------------
# 1. Define Model Parameters
# -----------------------------
N = np.array([6246073, 31530507, 13972160])  # (Children=0-19, Adults=20-49, Seniors=50+)
S_init = N.copy()
I_init = np.array([1, 1, 1])
R_init = np.array([0, 0, 0])
V_init = np.array([0, 0, 0])

# Combine into full initial condition vector: [S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3]
initial_conditions = np.array([
    S_init[0], I_init[0], R_init[0], V_init[0],
    S_init[1], I_init[1], R_init[1], V_init[1],
    S_init[2], I_init[2], R_init[2], V_init[2]
])

# Disease parameters
gamma = np.array([0.125, 0.083, 0.048])  # recovery rates
W = 0.005                               # waning immunity

# Vaccination thresholds
t1 = 30
t2 = 60

# Vacciantion ratio 
Vr0 = 0.01
Vr1 = 0.01
Vr2 = 0.01

# Transmission matrix
beta = np.array([
    [0.1901, 0.0660, 0.0028],
    [0.1084, 0.1550, 0.0453],
    [0.1356, 0.1540, 0.1165]
])

# -----------------------------
# Economic parameters
# -----------------------------
# Medical cost parameters from Kim et al. (example references)
cost_mild   = 160.8   # per day, mild
cost_hosp   = 432.5   # per day, hospitalized (non-ICU)
cost_icu    = 1129.5  # per day, ICU

# Fraction of infected who are mild/hospital/ICU for each age group
# 0-19:  95% mild, 4% hosp, 1% ICU
f_child_mild = 0.95
f_child_hosp = 0.04
f_child_icu  = 0.01

# 20-49: 90% mild, 8% hosp, 2% ICU
f_adult_mild = 0.90
f_adult_hosp = 0.08
f_adult_icu  = 0.02

# 50+:   70% mild, 20% hosp, 10% ICU
f_senior_mild = 0.70
f_senior_hosp = 0.20
f_senior_icu  = 0.10

# Wages, funeral, etc.
employment_rates = np.array([0.02, 0.694, 0.513])
daily_income     = np.array([27.74, 105.12, 92.45])
funeral_cost     = 9405.38
mortality_fraction = np.array([0.000001, 0.00003, 0.007])
vaccination_cost_per_capita = 35.76
GDP_per_capita = 31929

# GDP loss model: fraction = a * exp(b * beta_m) + c
a, b, c = -0.3648, -8.7989, -0.0012

time_span = 180
t = np.linspace(0, time_span, time_span)  # day 0..179

# -----------------------------
# 2. SIRV Model with Piecewise Vaccination
# -----------------------------
def deriv(y, t, N, beta, gamma, W, t1, t2):
    # piecewise vaccination rate
    if t < t1:
        v = Vr0
    elif t < t2:
        v = Vr1
    else:
        v = Vr2

    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y
    
    # Forces of infection
    lambda1 = beta[0, 0]*I1/N[0] + beta[0, 1]*I2/N[1] + beta[0, 2]*I3/N[2]
    lambda2 = beta[1, 0]*I1/N[0] + beta[1, 1]*I2/N[1] + beta[1, 2]*I3/N[2]
    lambda3 = beta[2, 0]*I1/N[0] + beta[2, 1]*I2/N[1] + beta[2, 2]*I3/N[2]
    
    dS1dt = -lambda1*S1 - v*S1 + W*R1 + W*V1
    dI1dt = lambda1*S1 - gamma[0]*I1
    dR1dt = gamma[0]*I1 - W*R1
    dV1dt = v*S1 - W*V1
    
    dS2dt = -lambda2*S2 - v*S2 + W*R2 + W*V2
    dI2dt = lambda2*S2 - gamma[1]*I2
    dR2dt = gamma[1]*I2 - W*R2
    dV2dt = v*S2 - W*V2
    
    dS3dt = -lambda3*S3 - v*S3 + W*R3 + W*V3
    dI3dt = lambda3*S3 - gamma[2]*I3
    dR3dt = gamma[2]*I3 - W*R3
    dV3dt = v*S3 - W*V3
    
    return [dS1dt, dI1dt, dR1dt, dV1dt,
            dS2dt, dI2dt, dR2dt, dV2dt,
            dS3dt, dI3dt, dR3dt, dV3dt]

def run_simulation(beta_mod):
    results = odeint(deriv, initial_conditions, t, args=(N, beta_mod, gamma, W, t1, t2))
    return results

# -----------------------------
# 3. Day-by-Day Cost Computation
# -----------------------------
def compute_cost_dataframe(results, beta_m, N, dt=1.0):
    """
    For each day i:
      - Extract I_children, I_adults, I_seniors
      - Compute daily medical cost based on mild/hosp/ICU fractions
      - Compute wage, death, vaccination, GDP costs
      - Accumulate all to get cumulative totals
    """
    T = results.shape[0]
    times = np.arange(T) * dt
    
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
    
    # daily GDP loss fraction
    GDP_loss_fraction = a * np.exp(b * beta_m) + c
    GDP_loss_rate = GDP_per_capita * abs(GDP_loss_fraction) * (np.sum(N)/365.0)
    
    for i in range(T):
        y = results[i]
        
        # Infecteds in each age group
        I_children = y[1]
        I_adults   = y[5]
        I_seniors  = y[9]
        
        # -----------------------------
        # 1) Medical costs
        #    Use severity splits for each age group
        # -----------------------------
        med_child = I_children * (
            f_child_mild*cost_mild + f_child_hosp*cost_hosp + f_child_icu*cost_icu
        )
        med_adult = I_adults * (
            f_adult_mild*cost_mild + f_adult_hosp*cost_hosp + f_adult_icu*cost_icu
        )
        med_senior = I_seniors * (
            f_senior_mild*cost_mild + f_senior_hosp*cost_hosp + f_senior_icu*cost_icu
        )
        
        med = med_child + med_adult + med_senior
        
        # -----------------------------
        # 2) Wage loss (already updated to handle caregiver + children’s small wage)
        # -----------------------------
        W_child  = daily_income[0]
        W_adult  = daily_income[1]
        W_senior = daily_income[2]
        
        E_child  = employment_rates[0]
        E_adult  = employment_rates[1]
        E_senior = employment_rates[2]
        
        wage_child_loss = W_child * E_child * I_children
        wage_caregiver_loss = W_adult * E_adult * I_children
        wage_adult_loss = W_adult * E_adult * I_adults
        wage_senior_loss= W_senior* E_senior* I_seniors
        
        wage = (wage_child_loss
                + wage_caregiver_loss
                + wage_adult_loss
                + wage_senior_loss)
        
        # -----------------------------
        # 3) Death cost
        # -----------------------------
        death = funeral_cost * (
            mortality_fraction[0]*I_children +
            mortality_fraction[1]*I_adults +
            mortality_fraction[2]*I_seniors
        )
        
        # -----------------------------
        # 4) Vaccination cost
        # -----------------------------
        V_children = y[3]
        V_adults   = y[7]
        V_seniors  = y[11]
        vacc = vaccination_cost_per_capita*(V_children + V_adults + V_seniors)
        
        # -----------------------------
        # 5) GDP loss
        # -----------------------------
        gdp_daily = GDP_loss_rate
        
        # -----------------------------
        # Sum of all daily costs
        # -----------------------------
        total_rate = med + wage + death + vacc + gdp_daily
        
        # record daily
        med_cost_arr[i]   = med
        wage_loss_arr[i]  = wage
        death_cost_arr[i] = death
        vacc_cost_arr[i]  = vacc
        gdp_loss_arr[i]   = gdp_daily
        total_cost_arr[i] = total_rate
        
        # accumulate
        if i == 0:
            cum_med_arr[i]   = med
            cum_wage_arr[i]  = wage
            cum_death_arr[i] = death
            cum_vacc_arr[i]  = vacc
            cum_gdp_arr[i]   = gdp_daily
            cum_total_arr[i] = total_rate
        else:
            cum_med_arr[i]   = cum_med_arr[i-1]   + med
            cum_wage_arr[i]  = cum_wage_arr[i-1]  + wage
            cum_death_arr[i] = cum_death_arr[i-1] + death
            cum_vacc_arr[i]  = cum_vacc_arr[i-1]  + vacc
            cum_gdp_arr[i]   = cum_gdp_arr[i-1]   + gdp_daily
            cum_total_arr[i] = cum_total_arr[i-1] + total_rate
    
    # Unpack compartments for reference
    S1_arr = results[:, 0]
    I1_arr = results[:, 1]
    R1_arr = results[:, 2]
    V1_arr = results[:, 3]
    
    S2_arr = results[:, 4]
    I2_arr = results[:, 5]
    R2_arr = results[:, 6]
    V2_arr = results[:, 7]
    
    S3_arr = results[:, 8]
    I3_arr = results[:, 9]
    R3_arr = results[:, 10]
    V3_arr = results[:, 11]
    
    df = pd.DataFrame({
        "time": times,
        "S1": S1_arr, "I1": I1_arr, "R1": R1_arr, "V1": V1_arr,
        "S2": S2_arr, "I2": I2_arr, "R2": R2_arr, "V2": V2_arr,
        "S3": S3_arr, "I3": I3_arr, "R3": R3_arr, "V3": V3_arr,
        # Daily costs
        "Med_Cost": med_cost_arr,
        "Wage_Loss": wage_loss_arr,
        "Death_Cost": death_cost_arr,
        "Vacc_Cost": vacc_cost_arr,
        "GDP_Loss": gdp_loss_arr,
        "Total_Rate": total_cost_arr,
        # Cumulative
        "Cumulative_Med_Cost": cum_med_arr,
        "Cumulative_Wage_Loss": cum_wage_arr,
        "Cumulative_Death_Cost": cum_death_arr,
        "Cumulative_Vacc_Cost": cum_vacc_arr,
        "Cumulative_GDP_Loss": cum_gdp_arr,
        "Cumulative_Cost": cum_total_arr
    })
    return df

# -----------------------------
# 4. Run Scenarios & Build Results
# -----------------------------
beta_multipliers = [1.0, 0.7, 0.35, 0.2]
results_list = []
cumulative_cost_dict = {}
cost_df_dict = {}

for multiplier in beta_multipliers:
    # 1) Run simulation
    beta_mod = beta * multiplier
    sim_results = run_simulation(beta_mod)
    
    # 2) Build daily cost DataFrame
    cost_df = compute_cost_dataframe(sim_results, beta_m=multiplier, N=N, dt=1.0)
    
    # store time-series for plotting
    cumulative_cost_dict[multiplier] = cost_df["Cumulative_Cost"].values
    cost_df_dict[multiplier] = cost_df
    
    # 3) Extract final (cumulative) values from the last row
    final_row = cost_df.iloc[-1]  # day 179
    final_cumulative_cost = final_row["Cumulative_Cost"]
    
    final_med   = final_row["Cumulative_Med_Cost"]
    final_wage  = final_row["Cumulative_Wage_Loss"]
    final_death = final_row["Cumulative_Death_Cost"]
    final_vacc  = final_row["Cumulative_Vacc_Cost"]
    final_gdp   = final_row["Cumulative_GDP_Loss"]
    
    # 4) Compute peak infected
    I_total = sim_results[:, 1] + sim_results[:, 5] + sim_results[:, 9]
    peak_value = np.max(I_total)
    peak_day   = t[np.argmax(I_total)]
    
    scenario_result = {
        "beta_multiplier": multiplier,
        "peak_infected": peak_value,
        "peak_day": peak_day,
        "Medical Cost": final_med,
        "Wage Loss": final_wage,
        "Death Cost": final_death,
        "Vaccination Cost": final_vacc,
        "GDP Loss": final_gdp,
        "Total Cost": final_cumulative_cost
    }
    results_list.append(scenario_result)

results_df = pd.DataFrame(results_list)
print("Day-by-Day Summation Results (time_span=180):")
print(results_df)

# -----------------------------
# 5. Plots & Outputs
# -----------------------------
# 5a. Grouped bar chart for final cumulative cost components
cost_components = ["Medical Cost", "Wage Loss", "Death Cost", "GDP Loss", "Vaccination Cost"]
n_groups = len(results_df)
index = np.arange(n_groups)
bar_width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))
for i, component in enumerate(cost_components):
    ax.bar(index + i*bar_width, results_df[component], bar_width, label=component)

ax.set_xlabel("Beta Multiplier")
ax.set_ylabel("Cumulative Cost (USD, Log Scale)")
ax.set_title("Cumulative Economic Cost by Beta Multiplier (Day-by-Day Integration)")
ax.set_xticks(index + bar_width*(len(cost_components)-1)/2)
ax.set_xticklabels(results_df["beta_multiplier"])
ax.legend()
ax.set_yscale("log")
ax.grid(True, which="both", ls="--")
plt.savefig("Figures/Economic_Cost_Breakdown_DaybyDay.png")
plt.show()

# 5b. Total cost vs. beta multiplier (log-scale)
plt.figure(figsize=(10, 6))
plt.plot(results_df["beta_multiplier"], results_df["Total Cost"],
         marker='o', linestyle='-', color='blue')
plt.xlabel("Beta Multiplier")
plt.ylabel("Final Cumulative Total Cost (USD, log scale)")
plt.title("Cumulative Total Cost vs. Beta Multiplier (Day-by-Day)")
plt.yscale("log")
plt.grid(True, which="both", ls="--")
plt.savefig("Figures/Total_Cost_vs_Beta_DaybyDay.png")
plt.show()

# 5c. Cumulative total cost vs. time for each scenario
plt.figure(figsize=(10, 6))
for multiplier in beta_multipliers:
    plt.plot(t, cumulative_cost_dict[multiplier], label=f"β={multiplier}")
plt.xlabel("Time (days)")
plt.ylabel("Cumulative Total Economic Cost (USD)")
plt.title("Cumulative Total Cost vs. Time (Day-by-Day Summation)")
plt.legend()
plt.grid(True, ls="--")
plt.savefig("Figures/Cumulative_Total_Cost_vs_Time_DaybyDay.png")
plt.show()

# 5d. Print final cost from each scenario
print("\nFinal Cumulative Cost from day-by-day data:")
for multiplier in beta_multipliers:
    final_cost = cost_df_dict[multiplier]["Cumulative_Cost"].iloc[-1]
    print(f"β={multiplier}: Final Cumulative Cost = {final_cost:,.2f} USD")

# Example of exporting one scenario's day-by-day DataFrame to CSV
selected_multiplier = 1.0
df_Lt = cost_df_dict[selected_multiplier]
df_Lt.to_csv(f"Figures/Cost_Detail_beta_{selected_multiplier}.csv", index=False)

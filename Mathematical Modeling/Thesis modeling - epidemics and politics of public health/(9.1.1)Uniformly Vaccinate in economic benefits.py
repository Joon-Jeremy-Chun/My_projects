# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 17:39:07 2025

@author: joonc

Day-by-day summation of cost components in an SIRV model with vaccination.
This version allows a constant vaccination rate v (ranging from 0.0 to 0.05)
with no social distancing intervention (i.e. β multiplier = beta_M). It computes the total economic cost,
cumulative infection-days, and then calculates the ICER (Incremental Cost-Effectiveness Ratio)
using the baseline (v = 0) as Plan A and each vaccination scenario as Plan B.
Plots are saved with filenames that include the simulation time span and a timestamp.
"""

import os
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from datetime import datetime

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

# Vaccination thresholds (not used now, replaced by constant vaccination rate)
t1 = 30
t2 = 60

# Transmission matrix
beta = np.array([
    [0.1901, 0.0660, 0.0028],
    [0.1084, 0.1550, 0.0453],
    [0.1356, 0.1540, 0.1165]
])

# -----------------------------
# Economic parameters
# -----------------------------
# Medical cost parameters
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

# Simulation time span (adjust as needed)
time_span = 150  # e.g., 150 days
t = np.linspace(0, time_span, time_span)  # days 0..time_span-1

# -----------------------------
# 2. SIRV Model with Constant Vaccination Rate
# -----------------------------
def deriv(y, t, N, beta, gamma, W, v):
    """
    SIRV model ODEs with constant vaccination rate v.
    """
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y
    
    lambda1 = beta[0,0]*I1/N[0] + beta[0,1]*I2/N[1] + beta[0,2]*I3/N[2]
    lambda2 = beta[1,0]*I1/N[0] + beta[1,1]*I2/N[1] + beta[1,2]*I3/N[2]
    lambda3 = beta[2,0]*I1/N[0] + beta[2,1]*I2/N[1] + beta[2,2]*I3/N[2]
    
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

def run_simulation(beta_mod, v):
    """
    Run simulation with baseline beta multiplier and constant vaccination rate v.
    """
    results = odeint(deriv, initial_conditions, t, args=(N, beta_mod, gamma, W, v))
    return results

# -----------------------------
# 3. Day-by-Day Cost Computation (Same as Base Code)
# -----------------------------
def compute_cost_dataframe(results, beta_m, N, dt=1.0):
    """
    For each day, compute:
      - Daily Medical, Wage, Death, Vaccination, and GDP costs,
      - Accumulate them to compute cumulative costs.
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
    
    # daily GDP loss fraction (based on beta_m)
    GDP_loss_fraction = a * np.exp(b * beta_m) + c
    GDP_loss_rate = GDP_per_capita * abs(GDP_loss_fraction) * (np.sum(N)/365.0)
    
    for i in range(T):
        y = results[i]
        
        # Infecteds in each age group
        I_children = y[1]
        I_adults   = y[5]
        I_seniors  = y[9]
        
        # 1) Medical Costs using severity splits for each age group
        med_child = I_children * (f_child_mild*cost_mild + f_child_hosp*cost_hosp + f_child_icu*cost_icu)
        med_adult = I_adults * (f_adult_mild*cost_mild + f_adult_hosp*cost_hosp + f_adult_icu*cost_icu)
        med_senior = I_seniors * (f_senior_mild*cost_mild + f_senior_hosp*cost_hosp + f_senior_icu*cost_icu)
        med = med_child + med_adult + med_senior
        
        # 2) Wage Loss (children's own wage + caregiver + adult and senior losses)
        W_child  = daily_income[0]
        W_adult  = daily_income[1]
        W_senior = daily_income[2]
        
        E_child  = employment_rates[0]
        E_adult  = employment_rates[1]
        E_senior = employment_rates[2]
        
        wage_child_loss = W_child * E_child * I_children
        wage_caregiver_loss = W_adult * E_adult * I_children
        wage_adult_loss = W_adult * E_adult * I_adults
        wage_senior_loss = W_senior * E_senior * I_seniors
        
        wage = wage_child_loss + wage_caregiver_loss + wage_adult_loss + wage_senior_loss
        
        # 3) Death Cost
        death = funeral_cost * (mortality_fraction[0]*I_children +
                                 mortality_fraction[1]*I_adults +
                                 mortality_fraction[2]*I_seniors)
        
        # 4) Vaccination Cost
        V_children = y[3]
        V_adults   = y[7]
        V_seniors  = y[11]
        vacc = vaccination_cost_per_capita * (V_children + V_adults + V_seniors)
        
        # 5) GDP Loss
        gdp_daily = GDP_loss_rate
        
        # Total Daily Cost
        total_rate = med + wage + death + vacc + gdp_daily
        
        med_cost_arr[i]   = med
        wage_loss_arr[i]  = wage
        death_cost_arr[i] = death
        vacc_cost_arr[i]  = vacc
        gdp_loss_arr[i]   = gdp_daily
        total_cost_arr[i] = total_rate
        
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

# -----------------------------
# 4. Run Vaccination Rate Scenarios (Baseline β = 1.0)
# -----------------------------
# We fix beta multiplier at 1.0 (no social distancing) and vary the constant vaccination rate
vaccination_rates = np.linspace(0.0, 0.05, 101)  # from 0.0 to 0.05 in 101 steps
results_vacc = []
cumulative_cost_vacc = []

# baseline beta (no SD intervention)
beta_M = 1.0
beta_mod = beta * beta_M

for v in vaccination_rates:
    sim_results = run_simulation(beta_mod, v)
    cost_df = compute_cost_dataframe(sim_results, beta_m=1.0, N=N, dt=1.0)
    
    # Get final cumulative cost (last day)
    final_cost = cost_df["Cumulative_Cost"].iloc[-1]
    results_vacc.append({"vaccination_rate": v, "Total_Cost": final_cost})
    cumulative_cost_vacc.append(final_cost)

results_vacc_df = pd.DataFrame(results_vacc)
print("Vaccination Rate Sensitivity Analysis Results:")
print(results_vacc_df)

# -----------------------------
# 5. Plot Total Economic Cost vs. Vaccination Rate
# -----------------------------
plt.figure(figsize=(10,6))
plt.plot(results_vacc_df["vaccination_rate"], results_vacc_df["Total_Cost"],
         marker='o', linestyle='-', color='green')
plt.xlabel("Vaccination Rate (constant daily)")
plt.ylabel("Total Economic Cost (USD)")
plt.title(f"Total Economic Cost vs. Vaccination Rate ({time_span} Days, β=1.0)")
plt.grid(True, ls="--")
plt.savefig(f"Figures/Total_Cost_vs_Vaccination_Rate_{time_span}days(beta{beta_M}).png")
plt.show()

# Find the optimal vaccination rate (minimizes total cost)
optimal_index = np.argmin(results_vacc_df["Total_Cost"])
optimal_rate = results_vacc_df["vaccination_rate"].iloc[optimal_index]
optimal_cost = results_vacc_df["Total_Cost"].iloc[optimal_index]

print(f"\nOptimal vaccination rate: {optimal_rate:.3f}")
print(f"Minimum total economic cost: ${optimal_cost:,.2f}")

# -----------------------------
# 6. Compute and Plot Peak Infected vs. Vaccination Rate
# -----------------------------
results_peak = []

# Use baseline beta (no SD intervention)
for v in vaccination_rates:
    sim_results = run_simulation(beta_mod, v)
    # Sum infected individuals across all age groups: I1 (children) + I2 (adults) + I3 (seniors)
    I_total = sim_results[:, 1] + sim_results[:, 5] + sim_results[:, 9]
    peak_value = np.max(I_total)
    results_peak.append({"vaccination_rate": v, "Peak_Infected": peak_value})

results_peak_df = pd.DataFrame(results_peak)
print("Peak Infected vs. Vaccination Rate Results:")
print(results_peak_df)

plt.figure(figsize=(10,6))
plt.plot(results_peak_df["vaccination_rate"], results_peak_df["Peak_Infected"],
         marker='o', linestyle='-', color='red')
plt.xlabel("Vaccination Rate (constant daily)")
plt.ylabel("Peak Infected Population")
plt.title(f"Peak Infected vs. Vaccination Rate ({time_span} Days, β={beta_M})")
plt.grid(True, ls="--")
plt.savefig(f"Figures/Peak_Infected_vs_Vaccination_Rate_{time_span}days.png")
plt.show()

# -----------------------------
# 7. Compute Total Cumulative Infection-Days vs. Vaccination Rate
# -----------------------------
cumulative_infection_days_vs_v = []

# For each vaccination rate, compute cumulative infection-days (integral of total infected)
for v in vaccination_rates:
    sim_results = run_simulation(beta_mod, v)
    # Total infected population at each time point
    I_total = sim_results[:, 1] + sim_results[:, 5] + sim_results[:, 9]
    # Since dt = 1 day, cumulative infection-days is simply the sum over time
    cumulative_infection_days = np.sum(I_total)
    cumulative_infection_days_vs_v.append(cumulative_infection_days)

# Create a DataFrame for cumulative infection-days vs. vaccination rate
cum_inf_days_df = pd.DataFrame({
    "vaccination_rate": vaccination_rates,
    "Cumulative_Infection_Days": cumulative_infection_days_vs_v
})
print("Cumulative Infection-Days vs. Vaccination Rate:")
print(cum_inf_days_df)

# Plot Total Cumulative Infection-Days vs. Vaccination Rate
plt.figure(figsize=(10,6))
plt.plot(cum_inf_days_df["vaccination_rate"], cum_inf_days_df["Cumulative_Infection_Days"],
         marker='o', linestyle='-', color='purple')
plt.xlabel("Vaccination Rate (constant daily)")
plt.ylabel("Total Cumulative Infection-Days")
plt.title(f"Total Cumulative Infection-Days vs. Vaccination Rate ({time_span} Days, β={beta_M})")
plt.grid(True, ls="--")
plt.savefig(f"Figures/Cumulative_InfectionDays_vs_VaccinationRate_{time_span}days(beta{beta_M}).png")
plt.show()

# -----------------------------
# 8. Compute ICER for Each Vaccination Scenario (Plan B) Relative to Baseline (Plan A: v = 0)
# -----------------------------
# Get baseline values (Plan A: vaccination_rate = 0)
baseline_cost = results_vacc_df.loc[results_vacc_df["vaccination_rate"] == 0, "Total_Cost"].values[0]
baseline_effect = cum_inf_days_df.loc[cum_inf_days_df["vaccination_rate"] == 0, "Cumulative_Infection_Days"].values[0]

icers = []
for index, row in results_vacc_df.iterrows():
    v = row["vaccination_rate"]
    cost = row["Total_Cost"]
    effect = cum_inf_days_df.loc[cum_inf_days_df["vaccination_rate"] == v, "Cumulative_Infection_Days"].values[0]
    
    # Incremental cost and incremental effect (infection-days averted)
    inc_cost = cost - baseline_cost
    inc_effect = baseline_effect - effect  # Note: lower infection-days is better
    
    # Avoid division by zero
    if inc_effect != 0:
        icer = inc_cost / inc_effect
    else:
        icer = np.nan
    icers.append(icer)

results_vacc_df["ICER"] = icers

print("Vaccination Rate Sensitivity Analysis with ICER:")
print(results_vacc_df)

# Plot ICER vs. Vaccination Rate
plt.figure(figsize=(10,6))
plt.plot(results_vacc_df["vaccination_rate"], results_vacc_df["ICER"],
         marker='o', linestyle='-', color='blue')
plt.xlabel("Vaccination Rate (constant daily)")
plt.ylabel("ICER (USD per infection-day averted)")
plt.title(f"ICER vs. Vaccination Rate ({time_span} Days, β={beta_M})")
plt.grid(True, ls="--")
plt.savefig(f"Figures/ICER_vs_Vaccination_Rate_{time_span}days(beta{beta_M}).png")
plt.show()

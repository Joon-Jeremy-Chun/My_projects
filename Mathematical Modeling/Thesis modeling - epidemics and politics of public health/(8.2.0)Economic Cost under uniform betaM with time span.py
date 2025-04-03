# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 06:24:57 2025

@author: joonc

Economic Cost Analysis for an Epidemiological Simulation with Log-Scale Bar Graphs.

This code simulates the epidemic under different social distancing measures (modeled
by scaling the transmission matrix with various β multipliers) and computes 
the associated economic costs, day-by-day.

Key change from the original:
    * We do a day-by-day summation of Medical, Wage, Death, and GDP costs.
    * The final bar chart uses the *cumulative* total from day 0 to day time_span.

We set `time_span = 180`. 
"""

import os
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Create directory to save figures if it doesn't exist
if not os.path.exists("Figures"):
    os.makedirs("Figures")

# -----------------------------
# 1. Load Parameters and Data
# -----------------------------
def load_parameters(file_path):
    """
    Loads parameters from an Excel file. 
    Adjust indexing if your file structure differs.
    """
    df = pd.read_excel(file_path, header=None)
    recovery_rates = df.iloc[4, 0:3].values      # gamma for each age group
    waning_immunity_rate = df.iloc[8, 0]         # W
    population_size = df.iloc[14, 0:3].values
    susceptible_init = df.iloc[14, 0:3].values
    infectious_init = df.iloc[16, 0:3].values  
    recovered_init = df.iloc[18, 0:3].values   
    vaccinated_init = df.iloc[20, 0:3].values  
    return (recovery_rates, waning_immunity_rate, 
            population_size, susceptible_init, 
            infectious_init, recovered_init, vaccinated_init)

# Adjust paths as needed
file_path = 'Inputs.xlsx'
(gamma, W, N, S_init, I_init, R_init, V_init) = load_parameters(file_path)

time_span = 30  # 180 days

# Load transmission matrix from CSV
beta_df = pd.read_csv('DataSets/Fitted_Beta_Matrix.csv')
beta = beta_df.iloc[0:3, 1:4].values

# Initial conditions: [S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3]
initial_conditions = [
    S_init[0], I_init[0], R_init[0], V_init[0],
    S_init[1], I_init[1], R_init[1], V_init[1],
    S_init[2], I_init[2], R_init[2], V_init[2]
]

# -----------------------------
# 2. SIRV Model (No Vaccination)
# -----------------------------
def deriv(y, t, N, beta, gamma, W):
    """
    SIRV model ODEs for three groups (Children, Adults, Seniors).
    Vaccination = 0 (V does not change).
    """
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y
    
    lambda1 = (beta[0,0]*I1/N[0] + beta[0,1]*I2/N[1] + beta[0,2]*I3/N[2])
    lambda2 = (beta[1,0]*I1/N[0] + beta[1,1]*I2/N[1] + beta[1,2]*I3/N[2])
    lambda3 = (beta[2,0]*I1/N[0] + beta[2,1]*I2/N[1] + beta[2,2]*I3/N[2])
    
    # Group 1 (Children)
    dS1dt = -lambda1*S1 + W*R1
    dI1dt = lambda1*S1 - gamma[0]*I1
    dR1dt = gamma[0]*I1 - W*R1
    dV1dt = 0
    
    # Group 2 (Adults)
    dS2dt = -lambda2*S2 + W*R2
    dI2dt = lambda2*S2 - gamma[1]*I2
    dR2dt = gamma[1]*I2 - W*R2
    dV2dt = 0
    
    # Group 3 (Seniors)
    dS3dt = -lambda3*S3 + W*R3
    dI3dt = lambda3*S3 - gamma[2]*I3
    dR3dt = gamma[2]*I3 - W*R3
    dV3dt = 0
    
    return [
        dS1dt, dI1dt, dR1dt, dV1dt,
        dS2dt, dI2dt, dR2dt, dV2dt,
        dS3dt, dI3dt, dR3dt, dV3dt
    ]

def run_simulation(beta_mod):
    t = np.linspace(0, time_span, time_span)  # daily steps: 0..179
    results = odeint(deriv, initial_conditions, t, args=(N, beta_mod, gamma, W))
    return results, t

# -----------------------------
# 3. Day-by-Day Economic Cost
# -----------------------------
# Economic parameters
M_mild = 160.8
employment_rates = np.array([0.02, 0.694, 0.513])  
daily_income = np.array([27.74, 105.12, 92.45])   
funeral_cost = 9405.38
mortality_fraction = np.array([0.000001, 0.00003, 0.007])
vaccination_cost_per_capita = 35.76  # Not used here (no vaccination)
GDP_per_capita = 31929
a, b, c = -0.3648, -8.7989, -0.0012  # for GDP loss fraction

def compute_cost_dataframe(results, beta_m, N, dt=1.0):
    """
    For each day:
      * Compute daily cost from Infectious compartments (Medical, Wage, Death),
      * Add daily GDP cost,
      * Sum & accumulate to get Cumulative_Cost.
    Returns a DataFrame with daily and cumulative columns.
    """
    T = results.shape[0]
    t_arr = np.arange(T)*dt
    
    # Arrays for daily cost
    med_cost_daily   = np.zeros(T)
    wage_loss_daily  = np.zeros(T)
    death_cost_daily = np.zeros(T)
    gdp_loss_daily   = np.zeros(T)
    total_daily      = np.zeros(T)
    
    # Arrays for cumulative
    cum_med_cost  = np.zeros(T)
    cum_wage_cost = np.zeros(T)
    cum_death_cost= np.zeros(T)
    cum_gdp_cost  = np.zeros(T)
    cum_total     = np.zeros(T)
    
    # GDP daily cost
    GDP_loss_fraction = a*np.exp(b*beta_m)+c
    GDP_loss_rate = abs(GDP_loss_fraction)*GDP_per_capita*(np.sum(N)/365.0)
    
    for i in range(T):
        # S1,I1,R1,V1, S2,I2,R2,V2, S3,I3,R3,V3
        y = results[i]
        I1, I2, I3 = y[1], y[5], y[9]
        
        # daily medical cost
        med_cost = M_mild * (I1 + I2 + I3)
        
        # daily wage loss
        #  (adults + seniors) + caregiving for children
        wage_adult_senior = (daily_income[1]*employment_rates[1]*I2 +
                             daily_income[2]*employment_rates[2]*I3)
        wage_caregiving   = daily_income[1]*employment_rates[1]*I1
        wage_loss         = wage_adult_senior + wage_caregiving
        
        # daily death cost from currently infected
        #  approximate: funeral cost * (mortality * I)
        death_cost = funeral_cost * (
            mortality_fraction[0]*I1 +
            mortality_fraction[1]*I2 +
            mortality_fraction[2]*I3
        )
        
        # daily GDP cost
        gdp_daily = GDP_loss_rate
        
        # sum
        total = med_cost + wage_loss + death_cost + gdp_daily
        
        # store daily
        med_cost_daily[i]   = med_cost
        wage_loss_daily[i]  = wage_loss
        death_cost_daily[i] = death_cost
        gdp_loss_daily[i]   = gdp_daily
        total_daily[i]      = total
        
        # accumulate
        if i == 0:
            cum_med_cost[i]   = med_cost
            cum_wage_cost[i]  = wage_loss
            cum_death_cost[i] = death_cost
            cum_gdp_cost[i]   = gdp_daily
            cum_total[i]      = total
        else:
            cum_med_cost[i]   = cum_med_cost[i-1]   + med_cost
            cum_wage_cost[i]  = cum_wage_cost[i-1]  + wage_loss
            cum_death_cost[i] = cum_death_cost[i-1] + death_cost
            cum_gdp_cost[i]   = cum_gdp_cost[i-1]   + gdp_daily
            cum_total[i]      = cum_total[i-1]      + total
    
    df = pd.DataFrame({
        "time" : t_arr,
        "S1"   : results[:,0],
        "I1"   : results[:,1],
        "R1"   : results[:,2],
        "V1"   : results[:,3],
        "S2"   : results[:,4],
        "I2"   : results[:,5],
        "R2"   : results[:,6],
        "V2"   : results[:,7],
        "S3"   : results[:,8],
        "I3"   : results[:,9],
        "R3"   : results[:,10],
        "V3"   : results[:,11],
        
        # Daily cost
        "Med_Daily"   : med_cost_daily,
        "Wage_Daily"  : wage_loss_daily,
        "Death_Daily" : death_cost_daily,
        "GDP_Daily"   : gdp_loss_daily,
        "Total_Daily" : total_daily,
        
        # Cumulative cost
        "Cumulative_Med"   : cum_med_cost,
        "Cumulative_Wage"  : cum_wage_cost,
        "Cumulative_Death" : cum_death_cost,
        "Cumulative_GDP"   : cum_gdp_cost,
        "Cumulative_Cost"  : cum_total
    })
    return df

# -----------------------------
# 4. Run Scenarios & Summaries
# -----------------------------
beta_multipliers = [1.0, 0.7, 0.35, 0.2]
results_list = []
cost_df_dict = {}

for multiplier in beta_multipliers:
    beta_mod = beta * multiplier
    
    sim_results, t = run_simulation(beta_mod)
    
    # Build day-by-day cost DataFrame
    cost_df = compute_cost_dataframe(sim_results, multiplier, N, dt=1.0)
    cost_df_dict[multiplier] = cost_df
    
    # Grab final cumulative costs from last row
    final_row = cost_df.iloc[-1]  # day 179
    med_final   = final_row["Cumulative_Med"]
    wage_final  = final_row["Cumulative_Wage"]
    death_final = final_row["Cumulative_Death"]
    gdp_final   = final_row["Cumulative_GDP"]
    total_final = final_row["Cumulative_Cost"]
    
    # Find peak infected (and day)
    I_tot = sim_results[:,1] + sim_results[:,5] + sim_results[:,9]
    peak_day = t[np.argmax(I_tot)]
    peak_val = np.max(I_tot)
    
    scenario_result = {
        "beta_multiplier": multiplier,
        "peak_infected": peak_val,
        "peak_day": peak_day,
        "Medical Cost": med_final,
        "Wage Loss": wage_final,
        "Death Cost": death_final,
        "GDP Loss": gdp_final,
        "Total Cost": total_final
    }
    results_list.append(scenario_result)

results_df = pd.DataFrame(results_list)
print("Scenario Results (Day-by-Day Summation):")
print(results_df)

# -----------------------------
# 5. Visualizations (Log Scale Bar Charts)
# -----------------------------
cost_components = ["Medical Cost", "Wage Loss", "Death Cost", "GDP Loss"]
n_groups = len(results_df)
index = np.arange(n_groups)
bar_width = 0.2

fig, ax = plt.subplots(figsize=(10,6))
for i, component in enumerate(cost_components):
    ax.bar(index + i*bar_width, results_df[component], bar_width, label=component)

ax.set_xlabel('Beta Multiplier')
ax.set_ylabel('Cost in USD (Log Scale)')
ax.set_title(f'Economic Cost Breakdown (Time Span: {time_span} days, Day-by-Day Summation)')
ax.set_xticks(index + bar_width*(len(cost_components)-1)/2)
ax.set_xticklabels(results_df['beta_multiplier'])
ax.legend()
ax.set_yscale('log')
ax.grid(True, which='both', ls='--')
plt.savefig(f"Figures/EconCost_Breakdown_DaybyDay_{time_span}days.png")
plt.show()

# 5b. Line Graph of Total Cost vs. Beta Multiplier
plt.figure(figsize=(10,6))
plt.plot(results_df["beta_multiplier"], results_df["Total Cost"], marker='o', linestyle='-', color='blue')
plt.xlabel("Beta Multiplier")
plt.ylabel("Total Economic Cost (USD, Log Scale)")
plt.title(f"Total Economic Cost vs Social Distancing (Day-by-Day, {time_span} days)")
plt.yscale('log')
plt.grid(True, which='both', ls='--')
plt.savefig(f"Figures/EconCost_vs_Beta_{time_span}days.png")
plt.show()

# Example: Print & Save a cost DataFrame for a chosen scenario
chosen_beta = 1.0
df_chosen = cost_df_dict[chosen_beta]
print(f"\nDetailed day-by-day cost for beta={chosen_beta}:")
print(df_chosen.head())

out_name = f"Figures/Cost_Detail_Beta_{chosen_beta}.csv"
df_chosen.to_csv(out_name, index=False)
print(f"Saved day-by-day costs to {out_name}")


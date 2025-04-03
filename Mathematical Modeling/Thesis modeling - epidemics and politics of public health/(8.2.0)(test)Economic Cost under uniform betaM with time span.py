# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 06:24:57 2025

@author: joonc

Economic Cost Analysis for an Epidemiological Simulation with Log-Scale Bar Graphs.

This code simulates the epidemic under different social distancing measures (modeled
by scaling the transmission matrix with various β multipliers) and computes 
the associated economic costs, day-by-day.

Key change:
  * Daily costs (Medical, Wage, Death, GDP) are computed and then cumulatively summed.
  * The final summary table (results_df) extracts the cumulative cost at the end of the simulation.
  
We set time_span = 180 days.
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
    Adjust the indices if your file structure differs.
    """
    df = pd.read_excel(file_path, header=None)
    recovery_rates = df.iloc[4, 0:3].values      # gamma for each age group
    waning_immunity_rate = df.iloc[8, 0]           # W
    population_size = df.iloc[14, 0:3].values
    susceptible_init = df.iloc[14, 0:3].values
    infectious_init = df.iloc[16, 0:3].values  
    recovered_init = df.iloc[18, 0:3].values   
    vaccinated_init = df.iloc[20, 0:3].values  
    return (recovery_rates, waning_immunity_rate, 
            population_size, susceptible_init, 
            infectious_init, recovered_init, vaccinated_init)

# Adjust file path as needed
file_path = 'Inputs.xlsx'
(gamma, W, N, S_init, I_init, R_init, V_init) = load_parameters(file_path)

time_span = 30  # 180 days

# Load transmission matrix from CSV (adjust path if needed)
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
    Vaccination is set to 0 (V remains constant).
    """
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y
    
    lambda1 = beta[0,0]*I1/N[0] + beta[0,1]*I2/N[1] + beta[0,2]*I3/N[2]
    lambda2 = beta[1,0]*I1/N[0] + beta[1,1]*I2/N[1] + beta[1,2]*I3/N[2]
    lambda3 = beta[2,0]*I1/N[0] + beta[2,1]*I2/N[1] + beta[2,2]*I3/N[2]
    
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
    
    return [dS1dt, dI1dt, dR1dt, dV1dt,
            dS2dt, dI2dt, dR2dt, dV2dt,
            dS3dt, dI3dt, dR3dt, dV3dt]

def run_simulation(beta_mod):
    t = np.linspace(0, time_span, time_span)  # daily steps: 0..time_span-1
    results = odeint(deriv, initial_conditions, t, args=(N, beta_mod, gamma, W))
    return results, t

# -----------------------------
# 3. Day-by-Day Economic Cost Computation
# -----------------------------
# Economic parameters
M_mild = 160.8
employment_rates = np.array([0.02, 0.694, 0.513])
daily_income = np.array([27.74, 105.12, 92.45])
funeral_cost = 9405.38
mortality_fraction = np.array([0.000001, 0.00003, 0.007])
vaccination_cost_per_capita = 35.76  # Vaccination is off here; this remains 0
GDP_per_capita = 31929
a, b, c = -0.3648, -8.7989, -0.0012

def compute_cost_dataframe(results, beta_m, N, dt=1.0):
    """
    For each day, compute:
      - Daily Medical, Wage, Death, and GDP costs.
      - Vaccination cost is 0.
      - Compute cumulative costs.
    Returns a DataFrame with daily and cumulative cost columns.
    """
    T = results.shape[0]
    t_arr = np.arange(T)*dt
    
    med_cost_daily   = np.zeros(T)
    wage_loss_daily  = np.zeros(T)
    death_cost_daily = np.zeros(T)
    vacc_cost_daily  = np.zeros(T)  # remains zero
    gdp_loss_daily   = np.zeros(T)
    total_daily      = np.zeros(T)
    
    cum_med_cost   = np.zeros(T)
    cum_wage_cost  = np.zeros(T)
    cum_death_cost = np.zeros(T)
    cum_vacc_cost  = np.zeros(T)
    cum_gdp_cost   = np.zeros(T)
    cum_total      = np.zeros(T)
    
    # Compute daily GDP loss rate (same each day)
    GDP_loss_fraction = a * np.exp(b * beta_m) + c
    GDP_loss_rate = abs(GDP_loss_fraction) * GDP_per_capita * (np.sum(N)/365.0)
    
    for i in range(T):
        y = results[i]
        I1, I2, I3 = y[1], y[5], y[9]
        
        med = M_mild * (I1 + I2 + I3)
        wage = (daily_income[1]*employment_rates[1]*I2 +
                daily_income[2]*employment_rates[2]*I3 +
                daily_income[1]*employment_rates[1]*I1)
        death = funeral_cost * (
            mortality_fraction[0]*I1 +
            mortality_fraction[1]*I2 +
            mortality_fraction[2]*I3
        )
        vacc = 0.0  # No vaccination
        
        gdp_daily = GDP_loss_rate
        
        total = med + wage + death + vacc + gdp_daily
        
        med_cost_daily[i] = med
        wage_loss_daily[i] = wage
        death_cost_daily[i] = death
        vacc_cost_daily[i] = vacc
        gdp_loss_daily[i] = gdp_daily
        total_daily[i] = total
        
        if i == 0:
            cum_med_cost[i] = med
            cum_wage_cost[i] = wage
            cum_death_cost[i] = death
            cum_vacc_cost[i] = vacc
            cum_gdp_cost[i] = gdp_daily
            cum_total[i] = total
        else:
            cum_med_cost[i] = cum_med_cost[i-1] + med
            cum_wage_cost[i] = cum_wage_cost[i-1] + wage
            cum_death_cost[i] = cum_death_cost[i-1] + death
            cum_vacc_cost[i] = cum_vacc_cost[i-1] + vacc
            cum_gdp_cost[i] = cum_gdp_cost[i-1] + gdp_daily
            cum_total[i] = cum_total[i-1] + total
            
    df = pd.DataFrame({
        "time": t_arr,
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
        # Daily costs
        "Med_Daily": med_cost_daily,
        "Wage_Daily": wage_loss_daily,
        "Death_Daily": death_cost_daily,
        "Vacc_Daily": vacc_cost_daily,
        "GDP_Daily": gdp_loss_daily,
        "Total_Daily": total_daily,
        # Cumulative costs
        "Cumulative_Med": cum_med_cost,
        "Cumulative_Wage": cum_wage_cost,
        "Cumulative_Death": cum_death_cost,
        "Cumulative_Vacc": cum_vacc_cost,
        "Cumulative_GDP": cum_gdp_cost,
        "Cumulative_Cost": cum_total
    })
    return df

# -----------------------------
# 4. Run Scenarios & Build Summary Table
# -----------------------------
beta_multipliers = [1.0, 0.7, 0.35, 0.2]
results_list = []
cost_df_dict = {}

for multiplier in beta_multipliers:
    beta_mod = beta * multiplier
    sim_results, t_vec = run_simulation(beta_mod)
    
    # Compute day-by-day cost DataFrame
    cost_df = compute_cost_dataframe(sim_results, multiplier, N, dt=1.0)
    cost_df_dict[multiplier] = cost_df
    
    # Extract final cumulative costs from the last day
    final_row = cost_df.iloc[-1]  # final day (day 179)
    med_final   = final_row["Cumulative_Med"]
    wage_final  = final_row["Cumulative_Wage"]
    death_final = final_row["Cumulative_Death"]
    vacc_final  = final_row["Cumulative_Vacc"]
    gdp_final   = final_row["Cumulative_GDP"]
    total_final = final_row["Cumulative_Cost"]
    
    # Compute peak infected and peak day from the day-by-day DataFrame
    I_tot = cost_df["I1"] + cost_df["I2"] + cost_df["I3"]
    peak_index = I_tot.idxmax()
    peak_day = cost_df.loc[peak_index, "time"]
    peak_val = I_tot.max()
    
    scenario_result = {
        "beta_multiplier": multiplier,
        "peak_infected": peak_val,
        "peak_day": peak_day,
        "Medical Cost": med_final,
        "Wage Loss": wage_final,
        "Death Cost": death_final,
        "Vaccination Cost": vacc_final,
        "GDP Loss": gdp_final,
        "Total Cost": total_final
    }
    results_list.append(scenario_result)

results_df = pd.DataFrame(results_list)

# -----------------------------
# 5. Print Summary Results
# -----------------------------
print("Time Span:", time_span, "days")
print("\nScenario Results (Cumulative Costs computed day-by-day):")
print(results_df)

print("\nDetailed Final Cumulative Costs per Beta Multiplier:")
for multiplier in beta_multipliers:
    df_temp = cost_df_dict[multiplier]
    final = df_temp.iloc[-1]
    # Compute peak infected from the daily DataFrame
    I_tot = df_temp["I1"] + df_temp["I2"] + df_temp["I3"]
    peak_index = I_tot.idxmax()
    peak_day = df_temp.loc[peak_index, "time"]
    peak_val = I_tot.max()
    print(f"\nBeta Multiplier: {multiplier}")
    print(f"  Peak Infected: {peak_val:,.2f}")
    print(f"  Peak day: {peak_day:,.2f}")
    print(f"  Cumulative Medical Cost: {final['Cumulative_Med']:,.2f}")
    print(f"  Cumulative Wage Loss:    {final['Cumulative_Wage']:,.2f}")
    print(f"  Cumulative Death Cost:   {final['Cumulative_Death']:,.2f}")
    print(f"  Cumulative Vaccination Cost: {final['Cumulative_Vacc']:,.2f}")
    print(f"  Cumulative GDP Loss:     {final['Cumulative_GDP']:,.2f}")
    print(f"  Cumulative Total Cost:   {final['Cumulative_Cost']:,.2f}")

# -----------------------------
# 6. Visualizations (Log Scale Bar Charts)
# -----------------------------
cost_components = ["Medical Cost", "Wage Loss", "Death Cost", "GDP Loss", "Vaccination Cost"]
n_groups = len(results_df)
index = np.arange(n_groups)
bar_width = 0.2

fig, ax = plt.subplots(figsize=(10,6))
for i, comp in enumerate(cost_components):
    ax.bar(index + i*bar_width, results_df[comp], bar_width, label=comp)

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

# 6b. Line Graph of Total Cost vs. Beta Multiplier
plt.figure(figsize=(10,6))
plt.plot(results_df["beta_multiplier"], results_df["Total Cost"], marker='o', linestyle='-', color='blue')
plt.xlabel("Beta Multiplier")
plt.ylabel("Total Economic Cost (USD, Log Scale)")
plt.title(f"Total Economic Cost vs Social Distancing (Day-by-Day, {time_span} days)")
plt.yscale("log")
plt.grid(True, which="both", ls="--")
plt.savefig(f"Figures/EconCost_vs_Beta_{time_span}days.png")
plt.show()

# 6c. Save Detailed Day-by-Day Cost DataFrame for a Selected Scenario (beta = 1.0)
selected_beta = 1.0
df_selected = cost_df_dict[selected_beta]
print(f"\nDetailed Day-by-Day Cost DataFrame for beta = {selected_beta}:")
print(df_selected.head())
out_file = f"Figures/Cost_Detail_Beta_{selected_beta}.csv"
df_selected.to_csv

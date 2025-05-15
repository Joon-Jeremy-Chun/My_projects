# -*- coding: utf-8 -*-
"""
Extended SIRV model with realistic daily vaccination capacity constraint.
Three simulation scenarios:
    1. Children prioritized initially
    2. Adults prioritized initially
    3. Seniors prioritized initially

For each, all available daily vaccine doses (e.g., 1% of total population) are assigned to one group for 60 days,
then evenly distributed across all groups.
"""

import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

# -----------------------------
# 1. LOAD PARAMETERS AND DATA
# -----------------------------
def load_parameters(file_path):
    df = pd.read_excel(file_path, header=None)
    recovery_rates = df.iloc[4, 0:3].values
    maturity_rates = df.iloc[6, 0:2].values
    waning_immunity_rate = df.iloc[8, 0]
    time_span = df.iloc[12, 0]
    population_size = df.iloc[14, 0:3].values
    susceptible_init = df.iloc[14, 0:3].values
    infectious_init = df.iloc[16, 0:3].values  
    recovered_init = df.iloc[18, 0:3].values   
    vaccinated_init = df.iloc[20, 0:3].values  
    return (recovery_rates, maturity_rates, waning_immunity_rate,
            time_span, population_size, susceptible_init,
            infectious_init, recovered_init, vaccinated_init)

# Load parameters from Excel file
file_path = 'Inputs.xlsx'
(gamma, mu, W, time_span, N, S_init, I_init, R_init, V_init) = load_parameters(file_path)

# Load transmission rates matrix from CSV file
transmission_rates_csv_path = 'DataSets/Fitted_Beta_Matrix.csv'
beta = pd.read_csv(transmission_rates_csv_path).iloc[0:3, 1:4].values

# Initial conditions for 3 groups (Children, Adults, Seniors)
initial_conditions = [
    S_init[0], I_init[0], R_init[0], V_init[0],
    S_init[1], I_init[1], R_init[1], V_init[1],
    S_init[2], I_init[2], R_init[2], V_init[2]
]

# -----------------------------
# 2. VACCINATION ALLOCATION FUNCTIONS
# -----------------------------
def allocate_vaccination_rates(ratio, capacity, N):
    """
    Calculate group-specific daily vaccination rates (v1, v2, v3)
    so that the total administered equals 'capacity' × total population.
    'ratio' gives the fraction of total capacity allocated to each group.
    """
    total_pop = N[0] + N[1] + N[2]
    v1 = ratio[0] * capacity * total_pop / N[0] if N[0] > 0 else 0
    v2 = ratio[1] * capacity * total_pop / N[1] if N[1] > 0 else 0
    v3 = ratio[2] * capacity * total_pop / N[2] if N[2] > 0 else 0
    return [v1, v2, v3]

# -----------------------------
# 3. TIME SEGMENTS & STRATEGY
# -----------------------------
t_a_end = 30      # Period a (days)
t_b_end = 60      # Period b
t_c_end = 180     # Period c
capacity = 0.01   # 1% of total population per day

# Scenarios: [children, adults, seniors] priority
ratio_children_priority = [1, 0, 0]
ratio_adults_priority   = [0, 1, 0]
ratio_seniors_priority  = [0, 0, 1]
ratio_all_equal         = [1/3, 1/3, 1/3]  # Even allocation after day 60

# Vaccination rates for each period and scenario
def get_vaccination_rates(period, priority):
    if period in ['a', 'b']:
        if priority == 'children':
            return allocate_vaccination_rates(ratio_children_priority, capacity, N)
        elif priority == 'adults':
            return allocate_vaccination_rates(ratio_adults_priority, capacity, N)
        elif priority == 'seniors':
            return allocate_vaccination_rates(ratio_seniors_priority, capacity, N)
    else:  # period c, d
        return allocate_vaccination_rates(ratio_all_equal, capacity, N)

def vaccination_strategy_factory(priority):
    """
    Returns a function a_t(t) that gives group vaccination rates at time t, for a given priority scenario.
    """
    def vaccination_strategy(t):
        if t < t_a_end:
            return get_vaccination_rates('a', priority)
        elif t < t_b_end:
            return get_vaccination_rates('b', priority)
        elif t < t_c_end:
            return get_vaccination_rates('c', priority)
        else:
            return get_vaccination_rates('d', priority)
    return vaccination_strategy

# -----------------------------
# 4. MODEL DEFINITION
# -----------------------------
def deriv(y, t, N, beta, gamma, mu, W, vaccination_strategy):
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y
    a_t = vaccination_strategy(t)
    lambda1 = beta[0, 0] * I1/N[0] + beta[0, 1] * I2/N[1] + beta[0, 2] * I3/N[2]
    lambda2 = beta[1, 0] * I1/N[0] + beta[1, 1] * I2/N[1] + beta[1, 2] * I3/N[2]
    lambda3 = beta[2, 0] * I1/N[0] + beta[2, 1] * I2/N[1] + beta[2, 2] * I3/N[2]
    dS1dt = -lambda1 * S1 - a_t[0] * S1 + W * R1 + W * V1
    dI1dt = lambda1 * S1 - gamma[0] * I1
    dR1dt = gamma[0] * I1 - W * R1
    dV1dt = a_t[0] * S1 - W * V1
    dS2dt = -lambda2 * S2 - a_t[1] * S2 + W * R2 + W * V2
    dI2dt = lambda2 * S2 - gamma[1] * I2
    dR2dt = gamma[1] * I2 - W * R2
    dV2dt = a_t[1] * S2 - W * V2
    dS3dt = -lambda3 * S3 - a_t[2] * S3 + W * R3 + W * V3
    dI3dt = lambda3 * S3 - gamma[2] * I3
    dR3dt = gamma[2] * I3 - W * R3
    dV3dt = a_t[2] * S3 - W * V3
    return [
        dS1dt, dI1dt, dR1dt, dV1dt,
        dS2dt, dI2dt, dR2dt, dV2dt,
        dS3dt, dI3dt, dR3dt, dV3dt
    ]

# -----------------------------
# 5. UTILITY: RUN SIMULATION AND PLOT
# -----------------------------
def run_example_and_plot(example_label, figure_title, vaccination_strategy, save_figure=True):
    t = np.linspace(0, time_span, int(time_span))
    results = odeint(deriv, initial_conditions, t, args=(N, beta, gamma, mu, W, vaccination_strategy))
    I_0_19 = results[:, 1]
    I_20_49 = results[:, 5]
    I_50_80 = results[:, 9]
    I_total = I_0_19 + I_20_49 + I_50_80

    # Metrics
    peak_idx_children = np.argmax(I_0_19)
    peak_children = I_0_19[peak_idx_children]
    time_peak_children = t[peak_idx_children]
    peak_idx_adults = np.argmax(I_20_49)
    peak_adults = I_20_49[peak_idx_adults]
    time_peak_adults = t[peak_idx_adults]
    peak_idx_seniors = np.argmax(I_50_80)
    peak_seniors = I_50_80[peak_idx_seniors]
    time_peak_seniors = t[peak_idx_seniors]
    peak_idx_total = np.argmax(I_total)
    peak_total = I_total[peak_idx_total]
    time_peak_total = t[peak_idx_total]
    print(f"Metrics for {example_label}:")
    print(f"  Children: Peak Infected = {peak_children:.2f} at day {time_peak_children:.2f}")
    print(f"  Adults:   Peak Infected = {peak_adults:.2f} at day {time_peak_adults:.2f}")
    print(f"  Seniors:  Peak Infected = {peak_seniors:.2f} at day {time_peak_seniors:.2f}")
    print(f"  Total:    Peak Infected = {peak_total:.2f} at day {time_peak_total:.2f}")
    print("-" * 50)

    # Plotting
    plt.figure(figsize=(12, 10))
    plt.suptitle(figure_title, fontsize=16, y=0.98)
    plt.subplot(2, 2, 1)
    plt.plot(t, I_0_19, label='I(t) Age 0-19')
    plt.xlabel('Time (days)')
    plt.ylabel('Infected Individuals')
    plt.title('Infections Age 0-19')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 2, 2)
    plt.plot(t, I_20_49, label='I(t) Age 20-49')
    plt.xlabel('Time (days)')
    plt.ylabel('Infected Individuals')
    plt.title('Infections Age 20-49')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 2, 3)
    plt.plot(t, I_50_80, label='I(t) Age 50-80+')
    plt.xlabel('Time (days)')
    plt.ylabel('Infected Individuals')
    plt.title('Infections Age 50-80+')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 2, 4)
    plt.plot(t, I_total, label='I(t) Total', color='black')
    plt.xlabel('Time (days)')
    plt.ylabel('Infected Individuals')
    plt.title('Total Infections')
    plt.legend()
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_figure:
        os.makedirs('Figures', exist_ok=True)
        figure_path = f'Figures/{example_label}.png'
        plt.savefig(figure_path, dpi=150)
        print(f"Saved figure to {figure_path}")
    plt.show()

# -----------------------------
# 6. RUN THREE EXAMPLES WITH CAPACITY CONSTRAINT
# -----------------------------
run_example_and_plot(
    example_label='Children Prioritized (capacity)',
    figure_title='Children Prioritized (capacity constraint, 1%)',
    vaccination_strategy=vaccination_strategy_factory('children')
)

run_example_and_plot(
    example_label='Adults Prioritized (capacity)',
    figure_title='Adults Prioritized (capacity constraint, 1%)',
    vaccination_strategy=vaccination_strategy_factory('adults')
)

run_example_and_plot(
    example_label='Seniors Prioritized (capacity)',
    figure_title='Seniors Prioritized (capacity constraint, 1%)',
    vaccination_strategy=vaccination_strategy_factory('seniors')
)

# -----------------------------
# 7. TOTAL INFECTIONS COMPARISON PLOT
# -----------------------------
def plot_total_infections_comparison():
    t = np.linspace(0, time_span, int(time_span))
    strategies = {
        'Children': vaccination_strategy_factory('children'),
        'Adults': vaccination_strategy_factory('adults'),
        'Seniors': vaccination_strategy_factory('seniors'),
    }
    results_dict = {}
    for label, strat in strategies.items():
        results = odeint(deriv, initial_conditions, t, args=(N, beta, gamma, mu, W, strat))
        I_total = results[:, 1] + results[:, 5] + results[:, 9]
        results_dict[label] = I_total
    plt.figure(figsize=(10, 6))
    for label, I_total in results_dict.items():
        plt.plot(t, I_total, label=f'{label} Prioritized')
    plt.xlabel('Time (days)')
    plt.ylabel('Total Infected Individuals')
    plt.title('Comparison of Total Infections (capacity constraint, 1%)')
    plt.legend()
    plt.grid(True)
    os.makedirs('Figures', exist_ok=True)
    figure_path = 'Figures/Total_Infections_Comparison_capacity.png'
    plt.savefig(figure_path, dpi=150)
    print(f"Saved comparison figure to {figure_path}")
    plt.show()

plot_total_infections_comparison()

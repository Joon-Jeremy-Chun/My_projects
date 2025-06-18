# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 21:17:48 2025

@author: Joon Chun

This script applies a 20 % reduction (multiplier = 0.8) to each individual
element of a 3×3 beta transmission matrix, records the total-infection peak,
and visualises the results.  Two figures are produced:

1) A 3×3 grid of time–series plots (one per beta element).
2) A 1×3 summary of the three beta elements that yield the largest
3) Baseline + Top-3 curves on a single axis, legend shows
     peak-reduction (%) and peak-time delay (Δt).

"""

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import argrelextrema

# ------------------------------------------------------------------
# Matplotlib global style tweaks
# ------------------------------------------------------------------
plt.rcParams.update({
    "font.size":        20,
    "axes.titlesize":   20,
    "axes.labelsize":   15,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":  15,
    "figure.titlesize": 30,
})

# Create output directory
os.makedirs("Figures", exist_ok=True)

# ------------------------------------------------------------------
# User-set parameters
# ------------------------------------------------------------------
reduction_factor = 0.2  # 20 % of original value

# ------------------------------------------------------------------
# Helper: load parameters from Excel
# ------------------------------------------------------------------
def load_parameters(xlsx_path):
    df = pd.read_excel(xlsx_path, header=None)

    recovery_rates    = df.iloc[4,  0:3].values
    maturity_rates    = df.iloc[6,  0:2].values
    waning_rate       = df.iloc[8,  0]
    vaccination_rates = df.iloc[10, 0:3].values
    time_span         = df.iloc[12, 0]
    population_size   = df.iloc[14, 0:3].values

    S_init = df.iloc[14, 0:3].values
    I_init = df.iloc[16, 0:3].values
    R_init = df.iloc[18, 0:3].values
    V_init = df.iloc[20, 0:3].values

    return (recovery_rates, maturity_rates, waning_rate,
            vaccination_rates, time_span, population_size,
            S_init, I_init, R_init, V_init)

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
xlsx_path = "Inputs.xlsx"
(gamma, mu, W,
 a, time_span, N,
 S_init, I_init, R_init, V_init) = load_parameters(xlsx_path)

beta = pd.read_csv("DataSets/Fitted_Beta_Matrix.csv").iloc[0:3, 1:4].values

# Initial state vector
y0 = [
    S_init[0], I_init[0], R_init[0], V_init[0],  # children
    S_init[1], I_init[1], R_init[1], V_init[1],  # adults
    S_init[2], I_init[2], R_init[2], V_init[2]   # seniors
]

# ------------------------------------------------------------------
# Age-structured SIRV model
# ------------------------------------------------------------------
def deriv(y, t, N, beta, gamma, mu, W, a):
    S1,I1,R1,V1, S2,I2,R2,V2, S3,I3,R3,V3 = y

    lam1 = beta[0,0]*I1/N[0] + beta[0,1]*I2/N[1] + beta[0,2]*I3/N[2]
    lam2 = beta[1,0]*I1/N[0] + beta[1,1]*I2/N[1] + beta[1,2]*I3/N[2]
    lam3 = beta[2,0]*I1/N[0] + beta[2,1]*I2/N[1] + beta[2,2]*I3/N[2]

    dS1 = -lam1*S1 - a[0]/N[0]*S1 + W*R1 + W*V1
    dI1 =  lam1*S1 - gamma[0]*I1 - mu[0]*I1
    dR1 =  gamma[0]*I1 - W*R1 - mu[0]*R1
    dV1 =  a[0]/N[0]*S1 - W*V1

    dS2 = -lam2*S2 + mu[0]*S1 - a[1]/N[1]*S2 + W*R2 + W*V2
    dI2 =  lam2*S2 + mu[0]*I1 - gamma[1]*I2 - mu[1]*I2
    dR2 =  gamma[1]*I2 + mu[0]*R1 - W*R2 - mu[1]*R2
    dV2 =  a[1]/N[1]*S2 - W*V2

    dS3 = -lam3*S3 + mu[1]*S2 - a[2]/N[2]*S3 + W*R3 + W*V3
    dI3 =  lam3*S3 + mu[1]*I2 - gamma[2]*I3
    dR3 =  gamma[2]*I3 + mu[1]*R2 - W*R3
    dV3 =  a[2]/N[2]*S3 - W*V3

    return dS1,dI1,dR1,dV1, dS2,dI2,dR2,dV2, dS3,dI3,dR3,dV3

# ------------------------------------------------------------------
# Simulation helpers
# ------------------------------------------------------------------
def simulate_modified_beta(i, j):
    """Return t, total-infected curve, peak value, peak time."""
    t = np.linspace(0, time_span, int(time_span))
    beta_mod = beta.copy()
    beta_mod[i, j] *= reduction_factor
    sol = odeint(deriv, y0, t, args=(N, beta_mod, gamma, mu, W, a))
    tot = sol[:,1] + sol[:,5] + sol[:,9]
    return t, tot, tot.max(), t[np.argmax(tot)]

# Baseline run (needed globally)
t = np.linspace(0, time_span, int(time_span))
sol_base = odeint(deriv, y0, t, args=(N, beta, gamma, mu, W, a))
baseline_total = sol_base[:,1] + sol_base[:,5] + sol_base[:,9]
baseline_peak  = baseline_total.max()
baseline_peak_time = t[np.argmax(baseline_total)]

# ------------------------------------------------------------------
# 1) Full 3×3 grid (optional)
# ------------------------------------------------------------------
def plot_component_wise_beta_reduction():
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            t_mod, curve, peak_val, peak_t = simulate_modified_beta(i, j)

            ax.plot(t_mod, curve, color='blue', label=f"Modified β[{i}][{j}]")
            ax.plot(t, baseline_total, '--', color='gray', label="Baseline")
            ax.scatter(peak_t, peak_val, color='red', zorder=5,
                       label=f"Peak: {peak_val:,.0f} @ t={peak_t:.1f}")
            ax.set_title(f"Reduction on β[{i}][{j}]")
            if j == 0:
                ax.set_ylabel("Total Infected")
            ax.set_xlabel("Time (days)")
            ax.legend(fontsize=9, loc="upper right")

    fig.suptitle("Total Infected Population (20 % component-wise β reductions)",
                 fontsize=22)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("Figures/component_wise_beta_reduction.png", dpi=300)
    plt.show()

# ------------------------------------------------------------------
# 2) 1×3 panel: separate axes for Top-3
# ------------------------------------------------------------------
def plot_top3_beta_reductions():
    info = {}
    for i in range(3):
        for j in range(3):
            info[(i,j)] = simulate_modified_beta(i, j)

    # keep top-3 by % reduction
    def pct_reduction(key):
        _, _, peak_val, _ = info[key]
        return (baseline_peak - peak_val) / baseline_peak * 100

    top3 = sorted(info.keys(), key=pct_reduction, reverse=True)[:3]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    colors = ['blue', 'orange', 'green']
    for ax, colour, (i,j) in zip(axes, colors, top3):
        t_mod, curve, peak_val, peak_t = info[(i,j)]
        pct = pct_reduction((i,j))
        ax.plot(t_mod, curve, color=colour, label=f"β[{i}][{j}]  ↓{pct:.1f}%")
        ax.plot(t, baseline_total, '--', color='gray', label="Baseline")
        ax.scatter(peak_t, peak_val, color='red', zorder=5)
        ax.set_title(f"Reduction on β[{i}][{j}]")
        ax.set_xlabel("Time (days)")
        if ax is axes[0]:
            ax.set_ylabel("Total Infected")
        ax.legend(fontsize=9, loc="upper right")

    fig.suptitle("Top-3 Component-wise β Reductions (20 %)", fontsize=22)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig("Figures/top3_beta_reductions.png", dpi=300)
    plt.show()

# ------------------------------------------------------------------
# 3) Single axis: baseline + Top-3 together, legend shows Δt
# ------------------------------------------------------------------
def plot_top3_combined():
    # Build dictionary { (i,j): (pct, curve, peak_val, peak_t) }
    info = {}
    for i in range(3):
        for j in range(3):
            t_mod, curve, peak_val, peak_t = simulate_modified_beta(i, j)
            pct_red = (baseline_peak - peak_val) / baseline_peak * 100
            info[(i,j)] = (pct_red, t_mod, curve, peak_val, peak_t)

    top3 = sorted(info.items(), key=lambda x: x[1][0], reverse=True)[:3]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(t, baseline_total, '--', color='gray', linewidth=2, label='Baseline')

    colors = ['blue', 'orange', 'green']
    for colour, ((i,j), (pct, t_mod, curve, peak_val, peak_t)) in zip(colors, top3):
        delay = peak_t - baseline_peak_time
        ax.plot(t_mod, curve, color=colour,
                label=f"β[{i}][{j}]  Peak↓{pct:.1f}% | Delay: {delay:+.0f} days")
        ax.scatter(peak_t, peak_val, color='red', zorder=5)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Total Infected")
    ax.set_title("Baseline vs. Top-3 Component-wise β Reductions (20 %)")
    ax.legend(fontsize=10)
    ax.set_ylim(0, baseline_total.max() * 1.25)

    plt.tight_layout()
    plt.savefig("Figures/top3_combined.png", dpi=300)
    plt.show()

# ------------------------------------------------------------------
# MAIN ─ choose which plots to create
# ------------------------------------------------------------------
# plot_component_wise_beta_reduction()  # full 3×3 grid (optional)
# plot_top3_beta_reductions()          # separate 1×3 panel (optional)
plot_top3_combined()                   # single-axis summary (default)



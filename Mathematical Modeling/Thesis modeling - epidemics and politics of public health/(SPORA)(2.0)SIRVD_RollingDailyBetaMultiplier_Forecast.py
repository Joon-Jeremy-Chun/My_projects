# -*- coding: utf-8 -*-
"""
(SPORA)(2.0) SIRVD_RollingDailyBetaMultiplier_Forecast

Recursive One-Step-Ahead Forecast with Daily 3x3 Beta Multiplier Optimization
Using solve_ivp (BDF method) with relaxed tolerances to mitigate stiffness issues.
Rolling period: March 1 to March 31, 2020.

Model note:
  • We use an age-structured SIRV model with three groups (0–19, 20–49, 50–80+).
  • Conceptually, infectious individuals may leave I_i either by recovery (γ_i)
    or disease-induced death (μ_i), feeding a cumulative death compartment D_i.
  • In this short-term rolling forecast, we set μ_i = 0 for all i, so numerically
    the dynamics are identical to the original SIRV model without an explicit
    death flow. The code is written, however, in the general form
        dI_i/dt = λ_i S_i − (γ_i + μ_i) I_i.

Produces:
  - 4 plots: one for each age group (0–19, 20–49, 50–80+) and one for the total infectious population.
  - 9 plots: one for each element of the 3x3 multiplier matrix over time.
  - 4 plots: forecast vs. actual using piecewise-constant mean multipliers (3 subperiods).
All figures are saved in the folder '(SPORA)figures'.
"""

import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Paths & output folder
# ---------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))

fig_folder = os.path.join(base_dir, "(SPORA)figures")
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)
print(f"Figures will be saved in: {fig_folder}")


def save_fig(fig, fname):
    """Save a figure to the SPORA figures folder and close it."""
    full_path = os.path.join(fig_folder, fname)
    print(f"Saving figure to: {full_path}")
    fig.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------
# 1. Define the SIRVD Model ODEs (D conceptual only)
# ---------------------------------------------------
def sirv_model(t, y, beta, gamma, a, delta, mu):
    """
    SIRV model with three age groups, allowing for a disease-induced
    death rate mu_i (conceptually feeding a cumulative D_i compartment).

    State ordering:
      y = [S1, I1, R1, V1,
           S2, I2, R2, V2,
           S3, I3, R3, V3]

    beta : 3x3 transmission rate matrix (used in the force of infection)
    gamma: array of length 3, recovery rates for each age group
    a    : array of length 3, vaccination rates for each age group
    delta: scalar, waning immunity rate (R_i, V_i -> S_i)
    mu   : array of length 3, disease-induced death rates
           (here we set mu = 0, so death has no numerical effect)
    """
    S1, I1, R1, V1, S2, I2, R2, V2, S3, I3, R3, V3 = y

    N1 = S1 + I1 + R1 + V1
    N2 = S2 + I2 + R2 + V2
    N3 = S3 + I3 + R3 + V3

    # Guard against division by zero
    N1 = max(N1, 1e-12)
    N2 = max(N2, 1e-12)
    N3 = max(N3, 1e-12)

    # Force of infection
    lambda1 = (
        beta[0, 0] * I1 / N1 +
        beta[0, 1] * I2 / N2 +
        beta[0, 2] * I3 / N3
    )
    lambda2 = (
        beta[1, 0] * I1 / N1 +
        beta[1, 1] * I2 / N2 +
        beta[1, 2] * I3 / N3
    )
    lambda3 = (
        beta[2, 0] * I1 / N1 +
        beta[2, 1] * I2 / N2 +
        beta[2, 2] * I3 / N3
    )

    # Group 1 (0–19)
    dS1 = -lambda1 * S1 - a[0] * S1 + delta * (R1 + V1)
    dI1 = lambda1 * S1 - (gamma[0] + mu[0]) * I1
    dR1 = gamma[0] * I1 - delta * R1
    dV1 = a[0] * S1 - delta * V1

    # Group 2 (20–49)
    dS2 = -lambda2 * S2 - a[1] * S2 + delta * (R2 + V2)
    dI2 = lambda2 * S2 - (gamma[1] + mu[1]) * I2
    dR2 = gamma[1] * I2 - delta * R2
    dV2 = a[1] * S2 - delta * V2

    # Group 3 (50–80+)
    dS3 = -lambda3 * S3 - a[2] * S3 + delta * (R3 + V3)
    dI3 = lambda3 * S3 - (gamma[2] + mu[2]) * I3
    dR3 = gamma[2] * I3 - delta * R3
    dV3 = a[2] * S3 - delta * V3

    # If we explicitly tracked D_i, we would add
    # dD1 = mu[0] * I1, dD2 = mu[1] * I2, dD3 = mu[2] * I3.
    # Here we only need S,I,R,V for fitting/forecasting.

    return [
        dS1, dI1, dR1, dV1,
        dS2, dI2, dR2, dV2,
        dS3, dI3, dR3, dV3
    ]


# ---------------------------------------------------
# 2. Load Real Data and Fixed Parameters
# ---------------------------------------------------
start_roll = pd.Timestamp('2020-03-01')
end_roll = pd.Timestamp('2020-04-01')

param_path = os.path.join(base_dir, "DataSets", "Korea_threeGroups_SIRV_parameter.csv")
inputs_path = os.path.join(base_dir, "Inputs.xlsx")
beta_path = os.path.join(base_dir, "DataSets", "Fitted_Beta_Matrix.csv")
mult_out_path = os.path.join(base_dir, "DataSets", "Daily_Beta_Multipliers_Rolling.csv")

roll_data = pd.read_csv(param_path)
roll_data["Date"] = pd.to_datetime(roll_data["Date"])
roll_data = roll_data[(roll_data["Date"] >= start_roll) & (roll_data["Date"] <= end_roll)].copy()
roll_data.sort_values(by="Date", inplace=True)
roll_data.reset_index(drop=True, inplace=True)
n_days = len(roll_data)
if n_days < 2:
    raise ValueError("Not enough data in the rolling period.")


def get_state_from_row(row):
    """Extract [S,I,R,V] for the three age groups from a row."""
    return np.array([
        row["S_0-19"],   row["I_0-19"],   row["R_0-19"],   row["V_0-19"],
        row["S_20-49"],  row["I_20-49"],  row["R_20-49"],  row["V_20-49"],
        row["S_50-80+"], row["I_50-80+"], row["R_50-80+"], row["V_50-80+"]
    ], dtype=float)


inputs = pd.read_excel(inputs_path, header=None)
gamma = inputs.iloc[4, 0:3].values
a = inputs.iloc[10, 0:3].values
delta = inputs.iloc[8, 0]

# Disease-induced death rates (set to zero in this short window)
mu = np.zeros(3)

optimal_beta_df = pd.read_csv(beta_path, index_col=0)
optimal_beta = optimal_beta_df.values

# ---------------------------------------------------
# 3. Rolling forecast settings
# ---------------------------------------------------
t_span = (0.0, 1.0)  # one-day integration window

# ---------------------------------------------------
# 4. Recursive Rolling Forecast (daily M optimization)
# ---------------------------------------------------
current_state = get_state_from_row(roll_data.iloc[0])
forecast_dates = []
predicted_states = []
daily_matrices = []

bounds = [(0.0, 2.0)] * 9
x0 = np.ones(9)

for d in range(n_days - 1):
    date_today = roll_data["Date"].iloc[d]
    date_next = roll_data["Date"].iloc[d + 1]
    print(f"\nDay {d+1}/{n_days}: optimizing M for {date_today} → {date_next}")

    target_state = get_state_from_row(roll_data.iloc[d + 1])

    def objective(x):
        """Objective function: squared error between 1-step forecast and target."""
        M = x.reshape((3, 3))
        beta_eff = M * optimal_beta
        sol = solve_ivp(
            lambda t, y: sirv_model(t, y, beta_eff, gamma, a, delta, mu),
            t_span, current_state,
            method="BDF", rtol=1e-3, atol=1e-6
        )
        if sol.y.shape[1] == 0:
            return 1e30
        final_state = sol.y[:, -1]
        return np.sum((final_state - target_state) ** 2)

    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)

    if (not res.success) or np.any(~np.isfinite(res.x)):
        print("  ⚠ Optimization failed or produced NaN. Using previous M or identity.")
        if len(daily_matrices) > 0 and np.all(np.isfinite(daily_matrices[-1])):
            M = daily_matrices[-1]
        else:
            M = np.eye(3)
    else:
        M = res.x.reshape((3, 3))

    daily_matrices.append(M)

    new_beta = M * optimal_beta
    sol = solve_ivp(
        lambda t, y: sirv_model(t, y, new_beta, gamma, a, delta, mu),
        t_span, current_state,
        method="BDF", rtol=1e-3, atol=1e-6
    )

    if sol.y.shape[1] == 0:
        forecast_state = current_state.copy()
    else:
        forecast_state = sol.y[:, -1]

    predicted_states.append(forecast_state)
    forecast_dates.append(date_next)

    current_state = forecast_state.copy()
    x0 = res.x if res.success and np.all(np.isfinite(res.x)) else np.ones(9)

# ---------------------------------------------------
# 5. Extract Forecasted Infectious Data
# ---------------------------------------------------
pred_I_0_19 = [state[1] for state in predicted_states]
pred_I_20_49 = [state[5] for state in predicted_states]
pred_I_50_80 = [state[9] for state in predicted_states]
pred_I_total = [state[1] + state[5] + state[9] for state in predicted_states]

actual_I_0_19 = roll_data["I_0-19"].iloc[1:].values
actual_I_20_49 = roll_data["I_20-49"].iloc[1:].values
actual_I_50_80 = roll_data["I_50-80+"].iloc[1:].values
actual_I_total = actual_I_0_19 + actual_I_20_49 + actual_I_50_80

# ---------------------------------------------------
# 6. Build DataFrame of Daily Optimal Multiplier Matrices
# ---------------------------------------------------
flat_matrices = []
for date, M in zip(roll_data["Date"].iloc[:-1], daily_matrices):
    flat_matrices.append([date] + list(M.flatten()))

cols = ["Date", "M00", "M01", "M02", "M10", "M11", "M12", "M20", "M21", "M22"]
multiplier_df = pd.DataFrame(flat_matrices, columns=cols)
multiplier_df["Date"] = pd.to_datetime(multiplier_df["Date"])

multiplier_df.to_csv(mult_out_path, index=False)
print(f"Saved daily multiplier matrices to: {mult_out_path}")

# ---------------------------------------------------
# 7. Plot: Infectious curves (forecast vs actual)
# ---------------------------------------------------
# Group 0–19
fig = plt.figure(figsize=(10, 6))
plt.plot(forecast_dates, pred_I_0_19, "ro-", label="Predicted I (0–19)")
plt.plot(forecast_dates, actual_I_0_19, "k--", label="Actual I (0–19)")
plt.title("Recursive Forecast for Age Group 0–19")
plt.xlabel("Date")
plt.ylabel("Infectious Population")
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()
save_fig(fig, "Forecast_I_0-19.png")

# Group 20–49
fig = plt.figure(figsize=(10, 6))
plt.plot(forecast_dates, pred_I_20_49, "ro-", label="Predicted I (20–49)")
plt.plot(forecast_dates, actual_I_20_49, "k--", label="Actual I (20–49)")
plt.title("Recursive Forecast for Age Group 20–49")
plt.xlabel("Date")
plt.ylabel("Infectious Population")
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()
save_fig(fig, "Forecast_I_20-49.png")

# Group 50–80+
fig = plt.figure(figsize=(10, 6))
plt.plot(forecast_dates, pred_I_50_80, "ro-", label="Predicted I (50–80+)")
plt.plot(forecast_dates, actual_I_50_80, "k--", label="Actual I (50–80+)")
plt.title("Recursive Forecast for Age Group 50–80+")
plt.xlabel("Date")
plt.ylabel("Infectious Population")
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()
save_fig(fig, "Forecast_I_50-80+.png")

# Total
fig = plt.figure(figsize=(10, 6))
plt.plot(forecast_dates, pred_I_total, "ro-", label="Predicted Total I(t)")
plt.plot(forecast_dates, actual_I_total, "k--", label="Actual Total I(t)")
plt.title("Recursive Forecast for Total Infectious Population")
plt.xlabel("Date")
plt.ylabel("Total Infectious Population")
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()
save_fig(fig, "Forecast_Total_I.png")

# ---------------------------------------------------
# 8. Plot: Multiplier time series with 3-level averages
# ---------------------------------------------------
t0 = start_roll
t1 = pd.Timestamp("2020-03-03")
t2 = pd.Timestamp("2020-03-10")
t3 = end_roll
subperiods = [t0, t1, t2, t3]

for col in cols[1:]:
    fig = plt.figure(figsize=(10, 4))
    plt.plot(multiplier_df["Date"], multiplier_df[col], marker="o", linestyle="-", label=col)

    for i in range(3):
        t_start = subperiods[i]
        t_end = subperiods[i + 1]
        if i < 2:
            mask = (multiplier_df["Date"] >= t_start) & (multiplier_df["Date"] < t_end)
        else:
            mask = (multiplier_df["Date"] >= t_start) & (multiplier_df["Date"] <= t_end)

        sub_data = multiplier_df.loc[mask, col]
        if not sub_data.empty:
            level_avg = sub_data.mean()
            plt.hlines(
                y=level_avg,
                xmin=t_start,
                xmax=t_end,
                colors="r",
                linestyles="--",
                label=f"Level {i+1}: {level_avg:.3f}"
            )

    plt.title(f"Time Series of {col} Over {len(multiplier_df)} Days")
    plt.xlabel("Date")
    plt.ylabel(col)
    plt.legend()
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    save_fig(fig, f"Multiplier_{col}.png")

# ---------------------------------------------------
# 9. Forecast using piecewise-constant mean multipliers (3 subperiods)
# ---------------------------------------------------
mean_matrices = []
for i in range(3):
    t_start = subperiods[i]
    t_end = subperiods[i + 1]
    if i < 2:
        mask = (multiplier_df["Date"] >= t_start) & (multiplier_df["Date"] < t_end)
    else:
        mask = (multiplier_df["Date"] >= t_start) & (multiplier_df["Date"] <= t_end)
    mean_values = multiplier_df.loc[mask, cols[1:]].mean().values
    mean_matrix = mean_values.reshape((3, 3))
    mean_matrices.append(mean_matrix)

pred_mean_I_0_19 = []
pred_mean_I_20_49 = []
pred_mean_I_50_80 = []
pred_mean_I_total = []
forecast_dates_mean = []

for i in range(3):
    t_start = subperiods[i]
    t_end = subperiods[i + 1]
    if i < 2:
        sub_dates = roll_data[(roll_data["Date"] >= t_start) & (roll_data["Date"] < t_end)]["Date"].tolist()
    else:
        sub_dates = roll_data[(roll_data["Date"] >= t_start) & (roll_data["Date"] <= t_end)]["Date"].tolist()

    if len(sub_dates) <= 1:
        continue

    sub_initial_state = get_state_from_row(roll_data[roll_data["Date"] == sub_dates[0]].iloc[0])
    current_state = sub_initial_state.copy()
    mean_beta = mean_matrices[i] * optimal_beta

    for d in range(1, len(sub_dates)):
        sol = solve_ivp(
            lambda t, y: sirv_model(t, y, mean_beta, gamma, a, delta, mu),
            t_span, current_state,
            method="BDF", rtol=1e-3, atol=1e-6
        )
        if sol.y.shape[1] == 0:
            forecast_state = current_state.copy()
        else:
            forecast_state = sol.y[:, -1]

        pred_mean_I_0_19.append(forecast_state[1])
        pred_mean_I_20_49.append(forecast_state[5])
        pred_mean_I_50_80.append(forecast_state[9])
        pred_mean_I_total.append(forecast_state[1] + forecast_state[5] + forecast_state[9])
        forecast_dates_mean.append(sub_dates[d])
        current_state = forecast_state.copy()

# Forecast vs actual using mean multipliers

# 0–19
fig = plt.figure(figsize=(10, 6))
actual_I_0_19_mean = roll_data.set_index("Date").loc[forecast_dates_mean, "I_0-19"].values
plt.plot(forecast_dates_mean, pred_mean_I_0_19, "ro-", label="Forecast using Mean Multiplier (0–19)")
plt.plot(forecast_dates_mean, actual_I_0_19_mean, "k--", label="Actual I (0–19)")
plt.title("Forecast (Using Mean Multipliers) for Age Group 0–19")
plt.xlabel("Date")
plt.ylabel("Infectious Population")
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()
save_fig(fig, "Forecast_Mean_I_0-19.png")

# 20–49
fig = plt.figure(figsize=(10, 6))
actual_I_20_49_mean = roll_data.set_index("Date").loc[forecast_dates_mean, "I_20-49"].values
plt.plot(forecast_dates_mean, pred_mean_I_20_49, "ro-", label="Forecast using Mean Multiplier (20–49)")
plt.plot(forecast_dates_mean, actual_I_20_49_mean, "k--", label="Actual I (20–49)")
plt.title("Forecast (Using Mean Multipliers) for Age Group 20–49")
plt.xlabel("Date")
plt.ylabel("Infectious Population")
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()
save_fig(fig, "Forecast_Mean_I_20-49.png")

# 50–80+
fig = plt.figure(figsize=(10, 6))
actual_I_50_80_mean = roll_data.set_index("Date").loc[forecast_dates_mean, "I_50-80+"].values
plt.plot(forecast_dates_mean, pred_mean_I_50_80, "ro-", label="Forecast using Mean Multiplier (50–80+)")
plt.plot(forecast_dates_mean, actual_I_50_80_mean, "k--", label="Actual I (50–80+)")
plt.title("Forecast (Using Mean Multipliers) for Age Group 50–80+")
plt.xlabel("Date")
plt.ylabel("Infectious Population")
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()
save_fig(fig, "Forecast_Mean_I_50-80+.png")

# Total
fig = plt.figure(figsize=(10, 6))
actual_I_total_mean = actual_I_0_19_mean + actual_I_20_49_mean + actual_I_50_80_mean
plt.plot(forecast_dates_mean, pred_mean_I_total, "ro-", label="Forecast using Mean Multiplier (Total)")
plt.plot(forecast_dates_mean, actual_I_total_mean, "k--", label="Actual Total I(t)")
plt.title("Forecast (Using Mean Multipliers) for Total Infectious Population")
plt.xlabel("Date")
plt.ylabel("Total Infectious Population")
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()
save_fig(fig, "Forecast_Mean_Total_I.png")

print("\nAll computations and figure generation completed successfully.")

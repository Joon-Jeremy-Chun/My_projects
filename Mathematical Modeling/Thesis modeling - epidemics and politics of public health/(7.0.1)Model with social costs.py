# -*- coding: utf-8 -*-
"""
Flow-based social-cost reference, completed 2026-10-20.

This new helper completes a former placeholder. It intentionally does not
replace the archived thesis/SPORA economic experiments or their saved results.
State order per group is S,I,R,V,D. Time units and cost-rate units must agree.
Vaccination is charged on administered-dose flow a_i*S_i, not the V_i stock.
"""

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid


def compute_social_costs(times, states, vaccination_rates, medical_rates,
                         wage_rates, cost_per_dose, cost_per_death,
                         gdp_loss_per_time=0.0):
    """Integrate costs on the actual time grid; return component totals.

    states has shape (n_times, 5*n_groups). Medical/wage rates are cost per
    infectious person per time unit. Vaccination rates are per susceptible
    person per time unit. D is cumulative deaths, so only new deaths within
    this observation window are charged. Scalars or per-group vectors are
    accepted for each rate. GDP loss is a nonnegative cost rate.
    """
    times = np.asarray(times, dtype=float)
    states = np.asarray(states, dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("times must be finite and strictly increasing.")
    if states.ndim != 2 or states.shape[0] != len(times) or states.shape[1] == 0 or states.shape[1] % 5:
        raise ValueError("states must have shape (n_times, 5*n_groups).")
    if not np.isfinite(states).all() or np.any(states < 0):
        raise ValueError("states must be finite and nonnegative.")
    groups = states.shape[1] // 5

    def rate(value, name):
        value = np.asarray(value, dtype=float)
        if value.ndim > 1:
            raise ValueError(f"{name} must be scalar or a per-group vector.")
        value = np.broadcast_to(value, (groups,))
        if not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f"{name} must be finite and nonnegative.")
        return value

    a = rate(vaccination_rates, "vaccination_rates")
    med = rate(medical_rates, "medical_rates")
    wage = rate(wage_rates, "wage_rates")
    dose = rate(cost_per_dose, "cost_per_dose")
    death = rate(cost_per_death, "cost_per_death")
    if not np.isfinite(gdp_loss_per_time) or gdp_loss_per_time < 0:
        raise ValueError("gdp_loss_per_time must be finite and nonnegative.")
    S, I, D = states[:, 0::5], states[:, 1::5], states[:, 4::5]
    if np.any(np.diff(D, axis=0) < -1e-10):
        raise ValueError("Cumulative deaths must not decrease.")

    def integral(values):
        return cumulative_trapezoid(values, times, initial=0)

    costs = pd.DataFrame({
        "time": times,
        "Medical_Cost": integral((I*med).sum(axis=1)),
        "Wage_Loss": integral((I*wage).sum(axis=1)),
        "Vaccination_Cost": integral((S*a*dose).sum(axis=1)),
        "Death_Cost": ((D-D[0])*death).sum(axis=1),
        "GDP_Loss": (times-times[0])*gdp_loss_per_time,
    })
    costs["Total_Cost"] = costs.drop(columns="time").sum(axis=1)
    return costs


def main():
    # A constant illustrative accounting fixture, not fitted epidemic data.
    times = np.array([0.0, 0.5, 2.0])
    states = np.tile([100.0, 2.0, 0.0, 0.0, 0.0], (3, 1))
    print(compute_social_costs(times, states, 0.01, 2.0, 3.0, 5.0, 10.0))


if __name__ == "__main__":
    main()


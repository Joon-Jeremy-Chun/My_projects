# -*- coding: utf-8 -*-
"""
Leaky-vaccine SIS reference implementation, completed 2026-10-20.

Normalized S, I, V populations; recovery returns to susceptibility. Vaccination
reduces susceptibility by efficacy in [0,1], and protection may wane. These
explicit assumptions complete the former 2024 placeholder; no empirical fit is
claimed. Rates are per model time unit.
"""

import numpy as np
from scipy.integrate import solve_ivp


def sis_leaky_rhs(t, y, beta=0.3, gamma=0.1, vaccination=0.05,
                  efficacy=0.7, waning=0.01):
    """Conserve S+I+V while allowing breakthrough infection and waning."""
    rates = np.asarray([beta, gamma, vaccination, waning], dtype=float)
    if not np.isfinite(rates).all() or np.any(rates < 0):
        raise ValueError("Rates must be finite and nonnegative.")
    if not np.isfinite(efficacy) or not 0 <= efficacy <= 1:
        raise ValueError("efficacy must be between 0 and 1.")
    susceptible, infectious, vaccinated = y
    infection_s = beta * susceptible * infectious
    infection_v = (1 - efficacy) * beta * vaccinated * infectious
    doses = vaccination * susceptible
    return np.array([
        -infection_s - doses + gamma * infectious + waning * vaccinated,
        infection_s + infection_v - gamma * infectious,
        doses - infection_v - waning * vaccinated,
    ])


def simulate(y0=(0.8, 0.2, 0.0), duration=100.0, points=501, **parameters):
    """Return a solve_ivp solution for a normalized, closed population."""
    y0 = np.asarray(y0, dtype=float)
    if y0.shape != (3,) or not np.isfinite(y0).all() or np.any(y0 < 0) or not np.isclose(y0.sum(), 1):
        raise ValueError("y0 must contain three nonnegative fractions summing to one.")
    if not np.isfinite(duration) or duration <= 0 or not isinstance(points, int) or points < 2:
        raise ValueError("duration must be positive and points an integer >= 2.")
    solution = solve_ivp(lambda t, y: sis_leaky_rhs(t, y, **parameters),
                         (0, duration), y0, t_eval=np.linspace(0, duration, points),
                         rtol=1e-8, atol=1e-10)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution


def main():
    import matplotlib.pyplot as plt
    solution = simulate()
    for values, label in zip(solution.y, ["Susceptible", "Infectious", "Vaccinated"]):
        plt.plot(solution.t, values, label=label)
    plt.xlabel("Model time")
    plt.ylabel("Population fraction")
    plt.title("SIS with imperfect vaccination: illustrative parameters")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()


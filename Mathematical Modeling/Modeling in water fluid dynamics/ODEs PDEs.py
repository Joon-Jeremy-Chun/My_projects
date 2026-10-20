# -*- coding: utf-8 -*-
"""
Viscous velocity diffusion: a small ODE/PDE reference, added 2026-10-20.

Complete the former placeholder with u_t = viscosity*u_xx on x in [0,1],
zero wall velocities, and u(x,0)=sin(pi*x). No pressure/advection term is
included. The analytical solution is sin(pi*x)*exp(-viscosity*pi**2*t).
This is a new educational example, not a reconstruction of an original result.
"""

import numpy as np
from scipy.integrate import solve_ivp


def exact_velocity(x, t, viscosity=0.1):
    return np.sin(np.pi * np.asarray(x)) * np.exp(-viscosity * np.pi**2 * t)


def solve_diffusion(intervals=40, duration=1.0, viscosity=0.1, points=51):
    """Discretize space centrally and integrate the interior ODE system."""
    if not isinstance(intervals, int) or intervals < 2:
        raise ValueError("intervals must be an integer >= 2.")
    if not np.isfinite([duration, viscosity]).all() or duration <= 0 or viscosity <= 0:
        raise ValueError("duration and viscosity must be finite and positive.")
    if not isinstance(points, int) or points < 2:
        raise ValueError("points must be an integer >= 2.")
    x = np.linspace(0, 1, intervals + 1)
    h = 1.0 / intervals

    def rhs(t, interior):
        u = np.pad(interior, (1, 1))
        return viscosity * (u[:-2] - 2*u[1:-1] + u[2:]) / h**2

    times = np.linspace(0, duration, points)
    solution = solve_ivp(rhs, (0, duration), np.sin(np.pi*x[1:-1]),
                         t_eval=times, method="BDF", rtol=1e-9, atol=1e-11)
    if not solution.success:
        raise RuntimeError(solution.message)
    velocity = np.zeros((intervals + 1, points))
    velocity[1:-1] = solution.y
    return x, solution.t, velocity


def main():
    import matplotlib.pyplot as plt
    x, times, velocity = solve_diffusion()
    plt.plot(x, velocity[:, -1], label="Central differences + BDF")
    plt.plot(x, exact_velocity(x, times[-1]), "--", label="Analytical sine mode")
    plt.xlabel("Normalized position")
    plt.ylabel("Normalized velocity")
    plt.title("Viscous diffusion between stationary walls")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

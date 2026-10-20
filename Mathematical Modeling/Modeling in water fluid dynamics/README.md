# Viscous Diffusion: From PDE to ODEs

**Can a spatial discretization recover a known velocity-diffusion solution?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../readme.md)

---

## Question and approach

The former placeholder now solves u_t=νu_xx between stationary walls, starting from sin(πx). Central spatial differences produce an interior ODE system integrated with BDF.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [ODEs PDEs.py](ODEs%20PDEs.py) | Defined helpers: `exact_velocity`, `solve_diffusion`, `main` |

## Inputs, tools, and outputs

Observed import roots: `matplotlib`, `numpy`, `scipy`. These observations are not a tested dependency lockfile.

## Interpretation and reproduction limits

This October 2026 reference excludes advection and pressure forcing. A two-grid regression checks convergence to the analytical decaying sine mode; it does not validate a general fluid solver.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

# Predator–Prey Dynamics

**How do two interacting populations evolve under the chosen nonlinear rates?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../readme.md)

---

## Question and approach

Read the Lotka–Volterra derivative, example rates and initial conditions, solve_ivp call, and then the phase-portrait variant.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [2x2 Lotka-Volterra equations.py](2x2%20Lotka-Volterra%20equations.py) | Defined helpers: `lotka_volterra` |
| [2x2 equations phase portratis.py](2x2%20equations%20phase%20portratis.py) | Defined helpers: `lotka_volterra` |

## Inputs, tools, and outputs

Observed import roots: `matplotlib`, `numpy`, `scipy`. These observations are not a tested dependency lockfile.

## Interpretation and reproduction limits

Parameters are illustrative. No population dataset or biological calibration is supplied for these runs.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

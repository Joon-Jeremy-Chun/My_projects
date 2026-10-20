# Differentiation, Quadrature, and Error Order

**How does mesh refinement change an approximation error?**

[Portfolio](../../../README.md) · [Evidence & results](../../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../../docs/REPAIRS.md) · [Reproduction](../../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../../docs/CODE_CATALOG.md) · [Collection overview](../../README.md)

---

## Question and approach

Compare finite-difference and quadrature formulas against the chosen references. The repaired Simpson study uses even interval counts, sin(x) on [0,π], and the analytical integral 2.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [Further Study-adaptive integration method.py](Further%20Study-adaptive%20integration%20method.py) | Defined helpers: `f`, `trapezoidal_rule`, `simpsons_rule`, `adaptive_quadrature` |
| [Numerical Differentiation.py](Numerical%20Differentiation.py) | Defined helpers: `FiveP`, `T_Sin` |
| [Numerical Integration-Finding Big O.py](Numerical%20Integration-Finding%20Big%20O.py) | Defined helpers: `SimC`, `convergence_study`, `main` |
| [Numerical Integration-Simpson's rule.py](Numerical%20Integration-Simpson%27s%20rule.py) | Defined helpers: `SimC`, `T_Sin` |
| [Numerical Integration-Visualiztion Big O and calucluation.py](Numerical%20Integration-Visualiztion%20Big%20O%20and%20calucluation.py) | Defined helpers: `T_Sin`, `integrand` |
| [Simpson_Rule.py](Simpson_Rule.py) | Defined helpers: `SimC` |
| [Summary-Differentiation-Five_step with Big O.py](Summary-Differentiation-Five_step%20with%20Big%20O.py) | Defined helpers: `five_step`, `g` |
| [Summary-Numerical Interation and finding Big O.py](Summary-Numerical%20Interation%20and%20finding%20Big%20O.py) | Defined helpers: `SimC`, `T_Sin` |

## Inputs, tools, and outputs

Observed import roots: `Simpson_Rule`, `math`, `matplotlib`, `numpy`, `scipy`. These observations are not a tested dependency lockfile.

## Interpretation and reproduction limits

Subinterval counts and node counts differ. Near machine precision, log-ratios are not stable order estimates; convergence evidence is tied to the actual selected test function.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../../docs/REPAIRS.md).

---

[Portfolio](../../../README.md) · [Evidence & results](../../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../../docs/REPAIRS.md) · [Reproduction](../../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../../docs/CODE_CATALOG.md)

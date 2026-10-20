# Root Finding and a Can-Design Example

**How can bisection and secant steps be combined on a bracket?**

[Portfolio](../../../README.md) · [Evidence & results](../../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../../docs/REPAIRS.md) · [Reproduction](../../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../../docs/CODE_CATALOG.md) · [Collection overview](../../README.md)

---

## Question and approach

The repaired hybrid function retains a sign-changing interval, safeguards secant steps, uses the true midpoint (a+b)/2, and raises on invalid inputs or iteration exhaustion. Other files preserve separate bisection/secant tables and plots.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [basic_bisection_method.py](basic_bisection_method.py) | Defined helpers: `bisection`, `f` |
| [basic_secant_method.py](basic_secant_method.py) | Defined helpers: `secant`, `f` |
| [hybrid_bisection+secant.py](hybrid_bisection%2Bsecant.py) | Defined helpers: `hybrid`, `f`, `g` |
| [hybrid_bisection+secant_example.py](hybrid_bisection%2Bsecant_example.py) | Defined helpers: `hybrid`, `f`, `g` |
| [save_in_dataframe_bisection_method.py](save_in_dataframe_bisection_method.py) | Defined helpers: `bisection`, `f` |
| [save_in_dataframe_scant_method.py](save_in_dataframe_scant_method.py) | Defined helpers: `secant`, `f` |
| [visualiztion_bisection_method.py](visualiztion_bisection_method.py) | Defined helpers: `bisection`, `f` |
| [visualiztion_scant_method.py](visualiztion_scant_method.py) | Defined helpers: `secant`, `f` |

## Inputs, tools, and outputs

Observed import roots: `math`, `matplotlib`, `numpy`, `pandas`. These observations are not a tested dependency lockfile.

2 preserved tracked visual artifacts are under [Numerical_root_finding/](.). Figure presence does not establish that a current run recreates that exact image.

## Interpretation and reproduction limits

The correction is checked on translated brackets, including a flat cubic near the root. Historical outputs are preserved; the can-design example is not a general certified optimizer.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../../docs/REPAIRS.md).

---

[Portfolio](../../../README.md) · [Evidence & results](../../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../../docs/REPAIRS.md) · [Reproduction](../../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../../docs/CODE_CATALOG.md)

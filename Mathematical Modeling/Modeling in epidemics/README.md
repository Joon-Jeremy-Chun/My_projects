# Foundational Epidemic ODE Models

**How do compartment and demographic choices change epidemic trajectories?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../readme.md)

---

## Question and approach

Start with SIR_model.py, compare normalized variants, and inspect the demographic SIR/SEIR and death-compartment experiments. Read each state ordering and incidence scaling independently.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [Normalized SIR-1part.py](Normalized%20SIR-1part.py) | Defined helpers: `deriv` |
| [Normalized SIR-2part.py](Normalized%20SIR-2part.py) | Defined helpers: `deriv` |
| [Normalized SIR-3part.py](Normalized%20SIR-3part.py) | Defined helpers: `deriv` |
| [Normalized SIR-4part.py](Normalized%20SIR-4part.py) | Defined helpers: `f_S`, `f_I`, `f_R`, `trapezoidal_pc` |
| [SEIR with demography.py](SEIR%20with%20demography.py) | Defined helpers: `deriv` |
| [SIR with demography.py](SIR%20with%20demography.py) | Defined helpers: `deriv` |
| [SIR-D-model.py](SIR-D-model.py) | Defined helpers: `deriv` |
| [SIR_model.py](SIR_model.py) | Defined helpers: `deriv` |
| [SIR_model_nomarlized.py](SIR_model_nomarlized.py) | Defined helpers: `deriv` |
| [SIR_model_normalized.py](SIR_model_normalized.py) | Defined helpers: `deriv` |
| [SIR_two group interaction (Children and Adults).py](SIR_two%20group%20interaction%20%28Children%20and%20Adults%29.py) | Defined helpers: `deriv` |

## Inputs, tools, and outputs

Observed import roots: `matplotlib`, `numpy`, `pandas`, `scipy`. These observations are not a tested dependency lockfile.

## Interpretation and reproduction limits

These are related experiments, not interchangeable implementations of one fitted model. Saved rates and initial populations should not be treated as current epidemiological estimates.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

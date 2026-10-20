# SIS Models with Vaccination

**How do vaccination, breakthrough infection, recovery, and waning change a closed population?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../readme.md)

---

## Question and approach

The complete-protection example now sends vaccination S→V and SIS recovery I→S. The completed leaky-vaccine file adds an efficacy parameter and waning, conserving S+I+V.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [SIS Model with Vaccination.py](SIS%20Model%20with%20Vaccination.py) | Defined helpers: `sis_model_vaccine` |
| [SIS Model with incomplete vaccine.py](SIS%20Model%20with%20incomplete%20vaccine.py) | Defined helpers: `sis_leaky_rhs`, `simulate`, `main` |

## Inputs, tools, and outputs

Observed import roots: `matplotlib`, `numpy`, `scipy`. These observations are not a tested dependency lockfile.

## Interpretation and reproduction limits

The 2026 completion defines explicit reference assumptions; it is not an empirical estimate of vaccine efficacy. Population-conservation and perfect-protection-limit regressions cover the new equations.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

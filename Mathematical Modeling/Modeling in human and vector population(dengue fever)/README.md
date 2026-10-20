# Human–Vector Transmission

**How can human infection and vector infection interact?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../readme.md)

---

## Question and approach

The four-state example couples S_h/I_h and S_m/I_m. The repaired mosquito infection term transfers susceptible vectors into infected vectors, and the modeled mortality removes both vector classes.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [Vector-borne diseases model.py](Vector-borne%20diseases%20model.py) | Defined helpers: `si_model` |

## Inputs, tools, and outputs

Observed import roots: `matplotlib`, `numpy`, `scipy`. These observations are not a tested dependency lockfile.

## Interpretation and reproduction limits

No vector recruitment is included, so vector population decreases at the chosen mortality rate. This is a qualitative reference, not a dengue forecasting model.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

# Continuous-Age Epidemic Experiments

**What changes when age and time are both represented on a grid?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../readme.md)

---

## Question and approach

Read the grid definitions, initial age profile, explicit transport/reaction updates, and age-boundary treatment. The separate two-group example now explicitly balances adult deaths and births.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [Logistic population stracture.py](Logistic%20population%20stracture.py) | Defined helpers: `logistic_function` |
| [ODEs_model.py](ODEs_model.py) | Defined helpers: `deriv` |
| [SIR-PDEs basic(W.O.Stability).py](SIR-PDEs%20basic%28W.O.Stability%29.py) | Defined helpers: `sigmoid` |
| [SIR-PDEs basic(test).py](SIR-PDEs%20basic%28test%29.py) | Defined helpers: `sigmoid` |
| [SIR-PDEs basic.py](SIR-PDEs%20basic.py) | Defined helpers: `sigmoid` |
| [SIR-PDEs extract(10,30,70).py](SIR-PDEs%20extract%2810%2C30%2C70%29.py) | Defined helpers: `sigmoid` |
| [SIR-PDEs with plots.py](SIR-PDEs%20with%20plots.py) | Top-level experiment; imports: matplotlib.pyplot, numpy |
| [SIR-PDEs_diff_age_time_steps_back&forw.py](SIR-PDEs_diff_age_time_steps_back%26forw.py) | Defined helpers: `sigmoid` |
| [SIR-population distribution.py](SIR-population%20distribution.py) | Top-level experiment; imports: matplotlib.pyplot, numpy |
| [example.py](example.py) | Defined helpers: `deriv` |
| [example2.py](example2.py) | Top-level experiment; imports: matplotlib.pyplot, numpy |

## Inputs, tools, and outputs

Observed import roots: `matplotlib`, `numpy`, `scipy`. These observations are not a tested dependency lockfile.

## Interpretation and reproduction limits

Boundary indexing, mesh stability, and positive/conservative solutions require checking per variant. The corrected two-group derivative is covered by a conservation test; the full collection is not thereby PDE-validated.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

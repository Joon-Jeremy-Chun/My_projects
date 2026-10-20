# Potential-Field Boundary Scenarios

**How do boundary values change a grid-based electric potential?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../README.md)

---

## Question and approach

Inspect the 100×100 grid, cloud/ground boundary values, in-place neighbor averaging, maximum iteration count, and successive-change tolerance. Compare the other two boundary scenarios.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [Electric potential PDEs-2nd scenario.py](Electric%20potential%20PDEs-2nd%20scenario.py) | Top-level experiment; imports: matplotlib.pyplot, numpy |
| [Electric potential PDEs-3rd scenario.py](Electric%20potential%20PDEs-3rd%20scenario.py) | Top-level experiment; imports: matplotlib.pyplot, numpy |
| [Electric potential PDEs-Finite Difference Moethod.py](Electric%20potential%20PDEs-Finite%20Difference%20Moethod.py) | Top-level experiment; imports: matplotlib.pyplot, numpy |

## Inputs, tools, and outputs

Observed import roots: `matplotlib`, `numpy`. These observations are not a tested dependency lockfile.

5 preserved tracked visual artifacts are under [Lightening Modeling/](.). Figure presence does not establish that a current run recreates that exact image.

## Interpretation and reproduction limits

In-place updates differ from simultaneous Jacobi updates. Max-iteration termination alone does not establish the requested tolerance was reached.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

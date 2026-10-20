# Gaussian Elimination and LU

**How are linear systems reduced and factored?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../README.md)

---

## Question and approach

Compare plain elimination, the partial-pivoting variant, and LU construction. Track row operations, pivot handling, and the chosen input system.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [Gaussian elimination.py](Gaussian%20elimination.py) | Defined helpers: `gaussian_elimination` |
| [Gussian elimination with partical pivoting.py](Gussian%20elimination%20with%20partical%20pivoting.py) | Defined helpers: `lu_decomposition_with_partial_pivoting` |
| [LU computation.py](LU%20computation.py) | Defined helpers: `lu_decomposition` |

## Inputs, tools, and outputs

Observed import roots: `numpy`. These observations are not a tested dependency lockfile.

## Interpretation and reproduction limits

A working small example does not establish robust behavior for singular, nearly singular, or badly scaled matrices.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

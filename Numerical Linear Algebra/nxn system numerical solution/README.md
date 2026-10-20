# Iterative Linear-System Variants

**How is a Jacobi iteration organized for concrete n×n examples?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../README.md)

---

## Question and approach

Compare the three small-system variants and inspect matrix dimensions, initialization, and the iteration loop.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [Jacobi method (3x3).py](Jacobi%20method%20%283x3%29.py) | Defined helpers: `jacobi_method` |
| [Jacobi method.py](Jacobi%20method.py) | Top-level experiment; imports: numpy |
| [example 3x3.py](example%203x3.py) | Top-level experiment; imports: numpy |

## Inputs, tools, and outputs

Observed import roots: `numpy`. These observations are not a tested dependency lockfile.

## Interpretation and reproduction limits

These are development examples, not a packaged solver with comprehensive input validation.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

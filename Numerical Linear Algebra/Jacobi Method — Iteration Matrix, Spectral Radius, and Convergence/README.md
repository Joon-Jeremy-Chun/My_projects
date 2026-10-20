# Jacobi Iteration and Spectral Radius

**How do the iteration matrix, residual, and actual error relate?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../README.md)

---

## Question and approach

The script forms D, L, U; builds C=D⁻¹(L+U); computes eigenvalues; and records Jacobi iterates. Inspect the currently active A rather than a commented alternative.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [Jacobi Method.py](Jacobi%20Method.py) | Defined helpers: `is_strictly_diagonally_dominant`, `jacobi` |

## Inputs, tools, and outputs

Observed import roots: `matplotlib`, `numpy`, `os`. These observations are not a tested dependency lockfile.

3 preserved tracked visual artifacts are under [figures/](figures). Figure presence does not establish that a current run recreates that exact image.

## Interpretation and reproduction limits

The stopping test uses successive iterates. Reported residual and error histories should be read separately, and a maximum-iteration exit is not convergence.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

# Least Squares and the Pseudoinverse

**How do several constructions solve the same overdetermined system?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../README.md)

---

## Question and approach

The 5×3 example compares numpy.linalg.lstsq, normal equations, and a manually constructed SVD pseudoinverse, including ranks and residual norms.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [Least-Squares and Pseudo-Inverse for an Overdetermined System.py](Least-Squares%20and%20Pseudo-Inverse%20for%20an%20Overdetermined%20System.py) | Top-level experiment; imports: matplotlib.pyplot, numpy, os |

## Inputs, tools, and outputs

Observed import roots: `matplotlib`, `numpy`, `os`. These observations are not a tested dependency lockfile.

## Interpretation and reproduction limits

Agreement on this matrix does not establish identical conditioning or numerical reliability of all methods on arbitrary data.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

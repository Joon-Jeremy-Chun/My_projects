# Power Iteration Examples

**How does repeated matrix multiplication reveal a dominant eigen-direction?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../README.md)

---

## Question and approach

Compare the 2×2 and 3×3 examples, including normalization, eigenvalue estimate, initial vector, and stopping behavior.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [Example, 2x2 matrics.py](Example%2C%202x2%20matrics.py) | Top-level experiment; imports: numpy |
| [Example, 3x3 matrics.py](Example%2C%203x3%20matrics.py) | Top-level experiment; imports: numpy |

## Inputs, tools, and outputs

Observed import roots: `numpy`. These observations are not a tested dependency lockfile.

## Interpretation and reproduction limits

Convergence depends on the spectrum and initialization. The file title alone is not a guarantee for every supplied matrix.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

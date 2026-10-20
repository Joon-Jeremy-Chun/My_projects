# Exact Probability and Matrix Arithmetic

**How can short scripts make a mathematical calculation inspectable?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../README.md)

---

## Question and approach

Fraction Computing.py uses Fraction for total-probability/Bayes arithmetic. Matrix multiply.py preserves a separate linear-algebra calculation.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [Fraction Computing.py](Fraction%20Computing.py) | Top-level experiment; imports: fractions |
| [Matrix multiply.py](Matrix%20multiply.py) | Defined helpers: `multiply_matrices` |

## Inputs, tools, and outputs

Observed import roots: `fractions`. These observations are not a tested dependency lockfile.

## Interpretation and reproduction limits

The guide covers the committed small examples. Larger local model-selection drafts are intentionally outside the publication scope.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

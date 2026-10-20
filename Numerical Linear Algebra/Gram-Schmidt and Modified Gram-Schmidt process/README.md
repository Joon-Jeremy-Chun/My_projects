# Classical and Modified Gram–Schmidt

**How do projection order and finite precision affect an orthogonal basis?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../README.md)

---

## Question and approach

Both functions return Q and R and display orthogonality measures. The classical function now projects each original input column; the modified function updates the remaining columns.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [Classical Gram-Schmidt&Modified Gram-Schmidt.py](Classical%20Gram-Schmidt%26Modified%20Gram-Schmidt.py) | Defined helpers: `classical_gram_schmidt`, `modified_gram_schmidt` |

## Inputs, tools, and outputs

Observed import roots: `matplotlib`, `numpy`, `os`. These observations are not a tested dependency lockfile.

1 preserved tracked visual artifacts are under [figures/](figures). Figure presence does not establish that a current run recreates that exact image.

## Interpretation and reproduction limits

The old classical implementation projected an already updated vector. Its stored chart predates the correction. A Hilbert-matrix regression checks reconstruction and distinct roundoff behavior.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

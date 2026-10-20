# Stochastic Lightning-Like Paths

**How can constrained random motion create a branching/path illustration?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../README.md)

---

## Question and approach

The representative model moves down/down-left/down-right on a finite grid until the bottom is reached. Compare the separate visual variants.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [Lightning modeling.py](Lightning%20modeling.py) | Defined helpers: `lightning_step` |
| [Simple thunder modeling.py](Simple%20thunder%20modeling.py) | Defined helpers: `lightning_step` |

## Inputs, tools, and outputs

Observed import roots: `matplotlib`, `numpy`. These observations are not a tested dependency lockfile.

## Interpretation and reproduction limits

This is a random-path illustration. It is not coupled to the separate potential-field solver and is not a calibrated discharge simulation.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

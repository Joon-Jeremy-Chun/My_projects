# Heron's Square-Root Iteration

**How does a square-root iterate approach its target from different initial guesses?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../README.md)

---

## Question and approach

Read the iterative update, termination parameters, and the regular/semilog/log-log error plots. Compare absolute and relative error definitions.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [Heron's method.py](Heron%27s%20method.py) | Defined helpers: `heron_sqrt` |

## Inputs, tools, and outputs

Observed import roots: `matplotlib`, `numpy`, `os`. These observations are not a tested dependency lockfile.

5 preserved tracked visual artifacts are under [figures/](figures). Figure presence does not establish that a current run recreates that exact image.

## Interpretation and reproduction limits

Interpret the chosen positive target and arithmetic precision. A log plot of one finite run is not a general numerical proof.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

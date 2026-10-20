# Conjugate Gradient: Exact and Floating Arithmetic

**What changes when the same small CG computation uses fractions or float64?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../README.md)

---

## Question and approach

Read the A=[[4,1],[1,3]], b=[1,-8], x0=[2,1] setup, the CG recurrence, and the rational/float comparison. The companion steepest-descent file supplies a different search direction example.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [Conjugate Gradient Method.py](Conjugate%20Gradient%20Method.py) | Defined helpers: `cg_method`, `print_cg_results`, `plot_cg_path` |
| [Gradient-Steepest Descent Only.py](Gradient-Steepest%20Descent%20Only.py) | Defined helpers: `gradient_method` |

## Inputs, tools, and outputs

Observed import roots: `fractions`, `matplotlib`, `numpy`, `os`. These observations are not a tested dependency lockfile.

3 preserved tracked visual artifacts are under [figures/](figures). Figure presence does not establish that a current run recreates that exact image.

## Interpretation and reproduction limits

The figures depict this 2×2 problem. They do not benchmark convergence or stability over arbitrary systems.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

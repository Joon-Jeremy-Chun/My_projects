# Series Convergence Diagnostics

**What do ratio, root, Raabe, Dirichlet, and Kummer examples look like numerically?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../README.md)

---

## Question and approach

Read the term-generating functions and finite index range. Compare test sequences, reference lines, and partial sums rather than using the saved plot alone.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [Raabe's vs other's.py](Raabe%27s%20vs%20other%27s.py) | Defined helpers: `double_factorial`, `a` |
| [Ratio vs root.py](Ratio%20vs%20root.py) | Defined helpers: `a` |
| [complex_convergence.py](complex_convergence.py) | Top-level experiment; imports: matplotlib.pyplot, numpy, os |
| [dirichlet_test_conditions.py](dirichlet_test_conditions.py) | Top-level experiment; imports: matplotlib.pyplot, numpy, os |
| [kummer_test_vs_others.py](kummer_test_vs_others.py) | Defined helpers: `double_factorial`, `a` |

## Inputs, tools, and outputs

Observed import roots: `math`, `matplotlib`, `numpy`, `os`. These observations are not a tested dependency lockfile.

4 preserved tracked visual artifacts are under [Figures/](Figures). Figure presence does not establish that a current run recreates that exact image.

## Interpretation and reproduction limits

Finite values and labeled limit lines are illustrative observations; asymptotic conclusions require the relevant mathematical hypotheses.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

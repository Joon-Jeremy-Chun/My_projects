# Transition-Matrix Worked Calculations

**How do initial distributions propagate and produce expectations?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../README.md)

---

## Question and approach

Read the transition-probability and expectation examples before numbered variants. The expectation script computes a distribution after three matrix multiplications and weights it by state values.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [1.py](1.py) | Top-level experiment; imports: matplotlib.pyplot, numpy |
| [2.py](2.py) | Defined helpers: `diffuse` |
| [3.py](3.py) | Top-level experiment; imports: matplotlib.pyplot, numpy |
| [4.py](4.py) | Defined helpers: `lightning_step`, `simulate_lightning` |
| [5.py](5.py) | Defined helpers: `sir_model` |
| [Digonaliztion.py](Digonaliztion.py) | Top-level experiment; imports: numpy |
| [Finding an expectation.py](Finding%20an%20expectation.py) | Top-level experiment; imports: numpy |
| [Transition probability matrix.py](Transition%20probability%20matrix.py) | Top-level experiment; imports: numpy |
| [eigenvalus eigenvectors.py](eigenvalus%20eigenvectors.py) | Top-level experiment; imports: sympy |

## Inputs, tools, and outputs

Observed import roots: `matplotlib`, `numpy`, `scipy`, `sympy`. These observations are not a tested dependency lockfile.

## Interpretation and reproduction limits

State order and row-/column-vector convention matter. The code is the preserved worked calculation, not an independently checked complete textbook answer key.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

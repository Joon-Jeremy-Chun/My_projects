# Boundary Values and Jacobi Iteration

**How is a two-endpoint boundary problem converted into a tridiagonal solve?**

[Portfolio](../../../README.md) · [Evidence & results](../../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../../docs/REPAIRS.md) · [Reproduction](../../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../../docs/CODE_CATALOG.md) · [Collection overview](../../README.md)

---

## Question and approach

The main example uses central differences for −u″+pu=qx(L−x), u(0)=u(L)=0, then iterates with an infinity-norm residual. Its right-hand-side sign now matches that stated equation.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [Animation (2nd senario) - Boundary Value Problem & the Jacobi Method.py](Animation%20%282nd%20senario%29%20-%20Boundary%20Value%20Problem%20%26%20the%20Jacobi%20Method.py) | Defined helpers: `jacobi`, `animate` |
| [Animation (3rd senario) - Boundary Value Problem & the Jacobi Method.py](Animation%20%283rd%20senario%29%20-%20Boundary%20Value%20Problem%20%26%20the%20Jacobi%20Method.py) | Defined helpers: `jacobi`, `animate` |
| [Animation - Boundary Value Problem & the Jacobi Method.py](Animation%20-%20Boundary%20Value%20Problem%20%26%20the%20Jacobi%20Method.py) | Defined helpers: `jacobi`, `animate` |
| [Boundary Value Problem & the Jacobi Method.py](Boundary%20Value%20Problem%20%26%20the%20Jacobi%20Method.py) | Defined helpers: `jacobi` |

## Inputs, tools, and outputs

Observed import roots: `matplotlib`, `numpy`. These observations are not a tested dependency lockfile.

## Interpretation and reproduction limits

The animation variants use a constant negative load and therefore represent a different forcing convention. They are preserved as variants, not described as identical copies of the repaired main equation.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../../docs/REPAIRS.md).

---

[Portfolio](../../../README.md) · [Evidence & results](../../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../../docs/REPAIRS.md) · [Reproduction](../../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../../docs/CODE_CATALOG.md)

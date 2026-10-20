# Solar-Pond Diffusion and Crank–Nicolson

**How can a forced diffusion equation be stepped implicitly?**

[Portfolio](../../../README.md) · [Evidence & results](../../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../../docs/REPAIRS.md) · [Reproduction](../../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../../docs/CODE_CATALOG.md) · [Collection overview](../../README.md)

---

## Question and approach

The two solar-pond scripts form A and B with their scheme scaled by two. Their source vector is now 2Δt·C(L−x), with the existing Dirichlet and zero-flux boundary terms.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [Example Jocobi Method.py](Example%20Jocobi%20Method.py) | Defined helpers: `jacobi`, `animate` |
| [Solar Pond - parabolic PDEs - Crank-Nicolson (code).py](Solar%20Pond%20-%20parabolic%20PDEs%20-%20Crank-Nicolson%20%28code%29.py) | Defined helpers: `crank_nicolson` |
| [Solar Pond - parabolic PDEs - Crank-Nicolson.py](Solar%20Pond%20-%20parabolic%20PDEs%20-%20Crank-Nicolson.py) | Defined helpers: `crank_nicolson` |

## Inputs, tools, and outputs

Observed import roots: `matplotlib`, `numpy`, `scipy`. These observations are not a tested dependency lockfile.

## Interpretation and reproduction limits

The parameter variants use different grids/time horizons. A manufactured steady-state regression checks spatial convergence after the source fix. No physical solar-pond calibration is claimed.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../../docs/REPAIRS.md).

---

[Portfolio](../../../README.md) · [Evidence & results](../../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../../docs/REPAIRS.md) · [Reproduction](../../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../../docs/CODE_CATALOG.md)

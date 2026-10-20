# Nonlinear Second-Order Phase Portraits

**How do parameter choices change the vector field of the chosen system?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../README.md)

---

## Question and approach

Read the right-hand side, phase-grid parameters, and plotted trajectories. The separate integration exercise is a related numerical exploration.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [(1.0.0)Phase Field Plot.py](%281.0.0%29Phase%20Field%20Plot.py) | Defined helpers: `plot_phase_field` |
| [ex)triple integrals.py](ex%29triple%20integrals.py) | Defined helpers: `S`, `integrand1`, `inner_y_ge_z`, `inner_y_le_z`, `inner_integral1`, `inner_z_le_y` … |

## Inputs, tools, and outputs

Observed import roots: `matplotlib`, `mpmath`, `numpy`, `os`. These observations are not a tested dependency lockfile.

5 preserved tracked visual artifacts are under [Figures/](Figures). Figure presence does not establish that a current run recreates that exact image.

## Interpretation and reproduction limits

A displayed finite vector field is not a full equilibrium/stability classification. Parameter-specific plots retain their original provenance.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

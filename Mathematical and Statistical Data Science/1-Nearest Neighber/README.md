# 1-NN Geometry and Dimensionality

**How does the nearest observation to the origin behave as dimension grows?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../README.md)

---

## Question and approach

The script samples in [-1,1]^p and uses f(x)=exp(-||x||²), with seed 42. The Monte Carlo settings are 2,000 observations, 300 replications, and dimensions 1–10. Panels show geometry, distance, MSE, squared bias, and variance.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [1NN.py](1NN.py) | Defined helpers: `f_value`, `sample_uniform`, `nearest_neighbor_to_origin`, `plot_panel_A`, `plot_panel_B`, `run_simulation` … |

## Inputs, tools, and outputs

Observed import roots: `matplotlib`, `numpy`, `os`, `pandas`. These observations are not a tested dependency lockfile.

4 preserved tracked visual artifacts are under [figures/](figures). Figure presence does not establish that a current run recreates that exact image.

## Interpretation and reproduction limits

The target is the fixed origin and the responses are noiseless. The observed error decomposition is specific to that simulation design.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

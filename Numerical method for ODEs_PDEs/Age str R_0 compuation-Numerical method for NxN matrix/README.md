# Next-Generation Matrices and Age Aggregation

**How do contact, population, and recovery assumptions feed a spectral-radius calculation?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../README.md)

---

## Question and approach

Start with the eight-band population/contact inputs, compare upsampling and smoothing, construct F and V, then inspect NGM=FV⁻¹ and eigenvalue/power/QR variants.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [8X8 given data set.py](8X8%20given%20data%20set.py) | Top-level experiment; imports: ace_tools, numpy, pandas |
| [F matrix-1.py](F%20matrix-1.py) | Top-level experiment; imports: matplotlib.pyplot, numpy, pandas, scipy.interpolate, scipy.ndimage |
| [F matrix.py](F%20matrix.py) | Top-level experiment; imports: matplotlib.pyplot, numpy, pandas, scipy.interpolate, scipy.ndimage |
| [NGM_and_R_0.py](NGM_and_R_0.py) | Defined helpers: `group_population` |
| [Numpy built-in R_0 computation.py](Numpy%20built-in%20R_0%20computation.py) | Defined helpers: `compute_spectral_radius` |
| [NxN contact matrix by Gaussian Smoothing-1.py](NxN%20contact%20matrix%20by%20Gaussian%20Smoothing-1.py) | Top-level experiment; imports: matplotlib.pyplot, numpy, pandas, scipy.ndimage |
| [NxN contact matrix by Gaussian Smoothing.py](NxN%20contact%20matrix%20by%20Gaussian%20Smoothing.py) | Top-level experiment; imports: matplotlib.pyplot, numpy, pandas, scipy.ndimage |
| [NxN contact matrix by polynomial interoplation.py](NxN%20contact%20matrix%20by%20polynomial%20interoplation.py) | Top-level experiment; imports: ace_tools, numpy, pandas, scipy.interpolate |
| [Power Method and QR Method.py](Power%20Method%20and%20QR%20Method.py) | Defined helpers: `power_method`, `compute_spectral_radius` |
| [Power_method_R_0 computation.py](Power_method_R_0%20computation.py) | Defined helpers: `power_method` |
| [US population 2020 with normalized ratios.py](US%20population%202020%20with%20normalized%20ratios.py) | Top-level experiment; imports: numpy, pandas, scipy.interpolate |
| [V&Vinverse matrix.py](V%26Vinverse%20matrix.py) | Top-level experiment; imports: numpy, pandas |
| [scipy built-in computation.py](scipy%20built-in%20computation.py) | Top-level experiment; imports: numpy, pandas, scipy.sparse.linalg, time |
| [visualiztion NGM.py](visualiztion%20NGM.py) | Top-level experiment; imports: matplotlib.pyplot, numpy |

## Inputs, tools, and outputs

Observed import roots: `ace_tools`, `matplotlib`, `numpy`, `pandas`, `scipy`, `time`. These observations are not a tested dependency lockfile.

## Interpretation and reproduction limits

The repaired NGM script now uses eight disjoint age bands containing all 18 population bins. Historical 800×800 CSVs are preserved and are not certified outputs of the repaired code.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

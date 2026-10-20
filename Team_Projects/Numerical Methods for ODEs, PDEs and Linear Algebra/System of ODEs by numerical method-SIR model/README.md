# SIR Systems and Predictor–Corrector Methods

**How can a coupled SIR model be advanced without a black-box integrator?**

[Portfolio](../../../README.md) · [Evidence & results](../../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../../docs/REPAIRS.md) · [Reproduction](../../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../../docs/CODE_CATALOG.md) · [Collection overview](../../README.md)

---

## Question and approach

The trapezoidal predictor–corrector example predicts with Euler, corrects with averaged derivatives, and plots the three compartments. Related scripts inspect critical contact values and alternative initial conditions.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [Finding critcal value - contect ratio plot.py](Finding%20critcal%20value%20-%20contect%20ratio%20plot.py) | Defined helpers: `dS`, `dI`, `dR` |
| [Finding critcal value - contect ratio.py](Finding%20critcal%20value%20-%20contect%20ratio.py) | Defined helpers: `dS`, `dI`, `dR` |
| [SIR case study.py](SIR%20case%20study.py) | Defined helpers: `f_S`, `f_I`, `f_R`, `trapezoidal_pc` |
| [SIR model by trapezoidal p-c method.py](SIR%20model%20by%20trapezoidal%20p-c%20method.py) | Defined helpers: `f_S`, `f_I`, `f_R`, `trapezoidal_pc` |
| [SIR model finding an equilibrium.py](SIR%20model%20finding%20an%20equilibrium.py) | Defined helpers: `f_S`, `f_I`, `f_R`, `trapezoidal_pc`, `find_equilibrium` |
| [Trapezoidal example.py](Trapezoidal%20example.py) | Defined helpers: `f`, `trapezoidal_pc` |
| [answer(a).py](answer%28a%29.py) | Defined helpers: `f_S`, `f_I`, `f_R`, `trapezoidal_pc` |
| [answer(b).py](answer%28b%29.py) | Defined helpers: `f_S`, `f_I`, `f_R`, `trapezoidal_pc` |
| [different I_0 values.py](different%20I_0%20values.py) | Defined helpers: `f_S`, `f_I`, `f_R`, `trapezoidal_pc` |
| [stopping criteria.py](stopping%20criteria.py) | Top-level experiment; imports: none detected |

## Inputs, tools, and outputs

Observed import roots: `math`, `matplotlib`, `numpy`, `pandas`, `scipy`. These observations are not a tested dependency lockfile.

## Interpretation and reproduction limits

Inspect whether states are proportions or counts before transferring beta values between experiments. This is a method exercise, not an independently fitted epidemic forecast.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../../docs/REPAIRS.md).

---

[Portfolio](../../../README.md) · [Evidence & results](../../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../../docs/REPAIRS.md) · [Reproduction](../../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../../docs/CODE_CATALOG.md)

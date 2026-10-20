# Initial-Value Problems and Error Studies

**How can a known solution help examine time-stepping accuracy?**

[Portfolio](../../../README.md) · [Evidence & results](../../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../../docs/REPAIRS.md) · [Reproduction](../../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../../docs/CODE_CATALOG.md) · [Collection overview](../../README.md)

---

## Question and approach

The representative test uses y′=2y/t+t²exp(t), y(1)=0, and y=t²(exp(t)−exp(1)). Inspect its two-step midpoint update and Euler startup before comparing endpoint errors.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [IVP-first order with h=0.01 abs error.py](IVP-first%20order%20with%20h%3D0.01%20abs%20error.py) | Defined helpers: `model`, `Mid_ptn`, `Exact_f` |
| [IVP-first order with order of accuracy-big O.py](IVP-first%20order%20with%20order%20of%20accuracy-big%20O.py) | Defined helpers: `model`, `Mid_ptn`, `Exact_f` |
| [IVP-first order-numerical solution with garphs.py](IVP-first%20order-numerical%20solution%20with%20garphs.py) | Defined helpers: `model`, `Mid_ptn`, `Exact_f` |
| [Tumor Research-IVPs - part 1 - ending behavior.py](Tumor%20Research-IVPs%20-%20part%201%20-%20ending%20behavior.py) | Defined helpers: `tumor` |
| [Tumor Research-IVPs - part 2 - finding a critical points.py](Tumor%20Research-IVPs%20-%20part%202%20-%20finding%20a%20critical%20points.py) | Defined helpers: `tumor` |

## Inputs, tools, and outputs

Observed import roots: `math`, `matplotlib`, `numpy`. These observations are not a tested dependency lockfile.

## Interpretation and reproduction limits

The source contains multiple scenarios and plotting/reference-line choices. A plotted slope alone does not validate every stepper or startup scheme.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../../docs/REPAIRS.md).

---

[Portfolio](../../../README.md) · [Evidence & results](../../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../../docs/REPAIRS.md) · [Reproduction](../../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../../docs/CODE_CATALOG.md)

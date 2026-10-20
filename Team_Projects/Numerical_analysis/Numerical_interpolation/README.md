# Interpolation, Least Squares, and Numerical Integration

**How do tabulated data and analytic functions lead to numerical estimates?**

[Portfolio](../../../README.md) · [Evidence & results](../../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../../docs/REPAIRS.md) · [Reproduction](../../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../../docs/CODE_CATALOG.md) · [Collection overview](../../README.md)

---

## Question and approach

The collection compares midpoint, Riemann, trapezoidal, and Simpson variants and includes a least-squares approximation study. Read the particular input definition before selecting a formula.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [Least_square_approximation.py](Least_square_approximation.py) | Top-level experiment; imports: matplotlib.pyplot, pandas |
| [Numerical Integration(with data points)_Midpoints.py](Numerical%20Integration%28with%20data%20points%29_Midpoints.py) | Defined helpers: `Mid` |
| [Numerical Integration(with data points)_Riemann Sum.py](Numerical%20Integration%28with%20data%20points%29_Riemann%20Sum.py) | Top-level experiment; imports: pandas |
| [Numerical Integration(with data points)_Simpson's Rule.py](Numerical%20Integration%28with%20data%20points%29_Simpson%27s%20Rule.py) | Defined helpers: `Sim` |
| [Numerical Integration(with data points)_Trapezoidal.py](Numerical%20Integration%28with%20data%20points%29_Trapezoidal.py) | Defined helpers: `Ta` |
| [Numerical Integration(with functions)_Riemann Sum.py](Numerical%20Integration%28with%20functions%29_Riemann%20Sum.py) | Defined helpers: `f`, `g`, `b`, `lrs` |
| [Numerical Integration(with functions)_Simpson's Rule.py](Numerical%20Integration%28with%20functions%29_Simpson%27s%20Rule.py) | Defined helpers: `Simf`, `f`, `g`, `my_function` |
| [Numerical Integration(with functions)_Trapezoidal.py](Numerical%20Integration%28with%20functions%29_Trapezoidal.py) | Defined helpers: `Trf`, `f`, `g` |
| [Numerical Integration_Midpoints.py](Numerical%20Integration_Midpoints.py) | Defined helpers: `Mid` |
| [Numerical Integration_Riemann Sum.py](Numerical%20Integration_Riemann%20Sum.py) | Defined helpers: `f`, `b`, `lrs` |
| [Numerical Integration_Simpson's Rule.py](Numerical%20Integration_Simpson%27s%20Rule.py) | Defined helpers: `Sim`, `Simf`, `f`, `g`, `my_function` |
| [Numerical Integration_Trapezoidal.py](Numerical%20Integration_Trapezoidal.py) | Defined helpers: `Ta` |
| [graph test.py](graph%20test.py) | Defined helpers: `my_function` |
| [plot_given_data.py](plot_given_data.py) | Top-level experiment; imports: matplotlib.pyplot, pandas |

## Inputs, tools, and outputs

Observed import roots: `matplotlib`, `numpy`, `pandas`. These observations are not a tested dependency lockfile.

1 preserved tracked visual artifacts are under [Numerical_interpolation/](.). Figure presence does not establish that a current run recreates that exact image.

## Interpretation and reproduction limits

Several filenames preserve parallel development variants. A correct reference formula does not establish every data-based branch has been freshly executed.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../../docs/REPAIRS.md).

---

[Portfolio](../../../README.md) · [Evidence & results](../../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../../docs/REPAIRS.md) · [Reproduction](../../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../../docs/CODE_CATALOG.md)

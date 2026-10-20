# Sequences, Series, and Zeta Experiments

**How can convergence and complex-function behavior be visualized?**

[Portfolio](../../../README.md) · [Evidence & results](../../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../../docs/REPAIRS.md) · [Reproduction](../../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../../docs/CODE_CATALOG.md) · [Collection overview](../../README.md)

---

## Question and approach

Compare the pointwise/uniformly labeled examples, series truncations, parameterized zeta evaluations, and functional-equation explorations.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [(Functional)Zeta function with two parameter functions.py](%28Functional%29Zeta%20function%20with%20two%20parameter%20functions.py) | Defined helpers: `generate_complex_inputs`, `compute_zeta_for_inputs`, `plot_results_multi` |
| [(Functional)Zeta function-functional equation.py](%28Functional%29Zeta%20function-functional%20equation.py) | Defined helpers: `zeta_functional_equation_rhs` |
| [(Functional)Zeta function-several zeros.py](%28Functional%29Zeta%20function-several%20zeros.py) | Defined helpers: `zeta_functional_equation_rhs` |
| [Pointwise - Uniform Convergence.py](Pointwise%20-%20Uniform%20Convergence.py) | Top-level experiment; imports: matplotlib.pyplot, numpy, pandas |
| [Pointwise Convergence.py](Pointwise%20Convergence.py) | Top-level experiment; imports: matplotlib.pyplot, numpy, pandas |
| [Uniform Convergence.py](Uniform%20Convergence.py) | Top-level experiment; imports: matplotlib.pyplot, numpy, pandas |
| [Uniform Convergence2.py](Uniform%20Convergence2.py) | Top-level experiment; imports: matplotlib.pyplot, numpy, pandas |
| [Zeta function in series.py](Zeta%20function%20in%20series.py) | Defined helpers: `zeta`, `plot_results`, `main` |
| [Zeta function vector addition idea.py](Zeta%20function%20vector%20addition%20idea.py) | Defined helpers: `partial_sums_zeta`, `plot_input_and_vectors`, `main` |
| [Zeta function with parameter functions zeta(lambda(t)).py](Zeta%20function%20with%20parameter%20functions%20zeta%28lambda%28t%29%29.py) | Defined helpers: `generate_complex_inputs`, `zeta`, `plot_results`, `main` |
| [Zeta function with two parameter functions.py](Zeta%20function%20with%20two%20parameter%20functions.py) | Defined helpers: `generate_complex_inputs`, `zeta`, `plot_results_multi`, `main` |

## Inputs, tools, and outputs

Observed import roots: `matplotlib`, `mpmath`, `numpy`, `pandas`. These observations are not a tested dependency lockfile.

## Interpretation and reproduction limits

The repeated directory name is preserved. Numerical samples do not prove uniform convergence, analytic continuation, or statements about all zeros.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../../docs/REPAIRS.md).

---

[Portfolio](../../../README.md) · [Evidence & results](../../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../../docs/REPAIRS.md) · [Reproduction](../../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../../docs/CODE_CATALOG.md)

# Random Walk and Monte Carlo Visualizations

**How do sampled increments accumulate into paths and endpoint distributions?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../README.md)

---

## Question and approach

Inspect the allowed steps and probabilities, then the number of steps and replications. One preserved 1D example runs 10,000 replications of 100 ±1 steps.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [Brownian Motion Animation.py](Brownian%20Motion%20Animation.py) | Defined helpers: `update`, `init` |
| [Brownian Motion Simulation- simple move.py](Brownian%20Motion%20Simulation-%20simple%20move.py) | Top-level experiment; imports: matplotlib.pyplot, numpy |
| [Normal distribution-1D random walk - Monte Carlo simulation - Animation.py](Normal%20distribution-1D%20random%20walk%20-%20Monte%20Carlo%20simulation%20-%20Animation.py) | Defined helpers: `update`, `init` |
| [Normal distribution-1D random walk - Monte Carlo simulation.py](Normal%20distribution-1D%20random%20walk%20-%20Monte%20Carlo%20simulation.py) | Top-level experiment; imports: matplotlib.pyplot, numpy |
| [Normal distribution-2D random walk final - 3D histogram.py](Normal%20distribution-2D%20random%20walk%20final%20-%203D%20histogram.py) | Top-level experiment; imports: matplotlib.pyplot, mpl_toolkits.mplot3d, numpy |
| [Normal distribution-2D random walk final - respect to X(2D histogram.py](Normal%20distribution-2D%20random%20walk%20final%20-%20respect%20to%20X%282D%20histogram.py) | Top-level experiment; imports: matplotlib.pyplot, numpy, seaborn |
| [Poisson distribution bar graph.py](Poisson%20distribution%20bar%20graph.py) | Top-level experiment; imports: matplotlib.pyplot, numpy, scipy.stats |
| [simple ligthning path.py](simple%20ligthning%20path.py) | Defined helpers: `simulate_lightning` |
| [test.py](test.py) | Defined helpers: `update`, `init` |

## Inputs, tools, and outputs

Observed import roots: `matplotlib`, `mpl_toolkits`, `numpy`, `scipy`, `seaborn`. These observations are not a tested dependency lockfile.

5 preserved tracked visual artifacts are under [Brownian Motion Modeling/](.). Figure presence does not establish that a current run recreates that exact image.

## Interpretation and reproduction limits

The discrete examples are not exact continuous Brownian trajectories. A historical comment referring to 100,000 runs differs from the actual 10,000-run assignment.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

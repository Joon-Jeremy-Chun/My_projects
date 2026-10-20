# Simple Markov Chain Examples

**How is a path generated from a state distribution and transition matrix?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../README.md)

---

## Question and approach

Start with the rainy/sunny two-state example, which samples 100 steps from a specified matrix. Compare Monte Carlo and Poisson/normal-distribution explorations afterward.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [2 space state example.py](2%20space%20state%20example.py) | Defined helpers: `simulate_markov_chain` |
| [Monte Carlo simulation for Markov chain.py](Monte%20Carlo%20simulation%20for%20Markov%20chain.py) | Top-level experiment; imports: matplotlib.pyplot, numpy |
| [Normal distribution-Monte Carlo simulation for Markov chain.py](Normal%20distribution-Monte%20Carlo%20simulation%20for%20Markov%20chain.py) | Top-level experiment; imports: matplotlib.pyplot, numpy |
| [Poisson distribution - Markov chain and Monte Carlo simulation.py](Poisson%20distribution%20-%20Markov%20chain%20and%20Monte%20Carlo%20simulation.py) | Top-level experiment; imports: matplotlib.pyplot, numpy |
| [Poisson distribution bar graph.py](Poisson%20distribution%20bar%20graph.py) | Top-level experiment; imports: matplotlib.pyplot, numpy, scipy.stats |
| [Simple thunder modeling.py](Simple%20thunder%20modeling.py) | Defined helpers: `lightning_step` |
| [piday_ipynb의_사본.py](piday_ipynb%EC%9D%98_%EC%82%AC%EB%B3%B8.py) | Defined helpers: `getAcc`, `simulate` |

## Inputs, tools, and outputs

Observed import roots: `matplotlib`, `numpy`, `scipy`. These observations are not a tested dependency lockfile.

## Interpretation and reproduction limits

Transition sampling and independently sampled distributions are different mechanisms. Several random experiments are unseeded; exact trajectories vary.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

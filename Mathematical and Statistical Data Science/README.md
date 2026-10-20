<p align="center"><img src="../assets/readme/statistical-learning.svg" alt="Mathematical &amp; Statistical Data Science" width="100%"></p>

# Mathematical & Statistical Data Science

**Nearest-neighbor geometry, dimensionality, and error decomposition.**

[Portfolio](../README.md) · [Evidence & results](../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../docs/REPAIRS.md) · [Reproduction](../docs/REPRODUCIBILITY.md) · [Full code catalog](../docs/CODE_CATALOG.md)

---

## At a glance

| Item | Description |
|---|---|
| Collection | Mathematical & Statistical Data Science |
| Subject source inventory | 1 Python · 0 R · 0 notebooks |
| Verification level | Source inspection, selected saved artifacts, and focused regressions described in the repair log. |

## Scope and interpretation

The preserved 1-NN study evaluates a fixed target at the origin with synthetic noiseless responses. It is not a trained real-world classifier or an independent generalization benchmark.

## Representative experiment

The preserved script studies **1-nearest-neighbor prediction at the origin** under a specified noiseless function. It links a geometric question—how close the nearest sample lies—to prediction error as dimension increases.

<p align="center"><img src="1-Nearest%20Neighber/figures/panel_D.png" alt="Archived 1-NN MSE, squared bias, and variance by dimension" width="850"></p>

*Saved Monte Carlo plot. The horizontal axis spans dimensions 1–10; the vertical quantities are function-value errors for this simulation. Seed, sample count, and replication count are in the linked study guide.*

The accompanying CSV and geometry panels preserve the experiment's numerical and visual artifacts. They were inspected as historical outputs, not recomputed to establish a new performance result.


## Project and study guides

| Guide | Question |
|---|---|
| [1-NN Geometry and Dimensionality](1-Nearest%20Neighber/README.md) | How does the nearest observation to the origin behave as dimension grows? |


## Reading and execution

Start with the guide matching your question, inspect its parameters and inputs, then follow links to source. The order is an explanatory reading path; it is not a guarantee that every script executes consecutively.

Read [the reproduction guide](../docs/REPRODUCIBILITY.md) before running code. Source files can load external data, contact services, write artifacts, run large searches, or control hardware. The focused validation suite does not reproduce all historical experiments.

---

[Portfolio](../README.md) · [Evidence & results](../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../docs/REPAIRS.md) · [Reproduction](../docs/REPRODUCIBILITY.md) · [Full code catalog](../docs/CODE_CATALOG.md)

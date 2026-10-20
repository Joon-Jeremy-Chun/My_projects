# Discrete Diffusion Operators

**What does an iteration operator reveal about one diffusion discretization?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../README.md)

---

## Question and approach

The example constructs small A and B matrices, forms A⁻¹B, and inspects eigenvalues and diagonalization.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [inf_mat computation with 0 boundary.py](inf_mat%20computation%20with%200%20boundary.py) | Top-level experiment; imports: numpy |
| [inf_mat with 0 boundary_proof by eigenvalues.py](inf_mat%20with%200%20boundary_proof%20by%20eigenvalues.py) | Top-level experiment; imports: numpy |

## Inputs, tools, and outputs

Observed import roots: `numpy`. These observations are not a tested dependency lockfile.

## Interpretation and reproduction limits

The numerical matrix calculation is distinct from a proof for all meshes and boundary conditions.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

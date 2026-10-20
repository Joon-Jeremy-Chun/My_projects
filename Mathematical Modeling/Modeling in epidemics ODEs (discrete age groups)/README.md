# Discrete Age-Group Epidemic Models

**How can contact between children, adults, and seniors be represented in ODEs?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../readme.md)

---

## Question and approach

Inspect the SIR_i derivative and the 3×3 contact matrix, then compare two-/three-group variants and their 2D/3D views. The representative source credits work by Vishaal Ram and Laura P. Schaposnik.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [Age-contact matrix computation.py](Age-contact%20matrix%20computation.py) | Top-level experiment; imports: numpy |
| [SIR_i groups interaction with 3D plots.py](SIR_i%20groups%20interaction%20with%203D%20plots.py) | Defined helpers: `deriv` |
| [SIR_tree groups interaction with 3D plots.py](SIR_tree%20groups%20interaction%20with%203D%20plots.py) | Defined helpers: `deriv` |
| [SIR_tree groups-example.py](SIR_tree%20groups-example.py) | Defined helpers: `deriv` |
| [SIR_two group interaction (Children and Adults).py](SIR_two%20group%20interaction%20%28Children%20and%20Adults%29.py) | Defined helpers: `deriv` |
| [SIR_two group interaction_numerical (Children and Adults).py](SIR_two%20group%20interaction_numerical%20%28Children%20and%20Adults%29.py) | Defined helpers: `deriv` |

## Inputs, tools, and outputs

Observed import roots: `matplotlib`, `mpl_toolkits`, `numpy`, `scipy`. These observations are not a tested dependency lockfile.

3 preserved tracked visual artifacts are under [Modeling in epidemics ODEs (discrete age groups)/](.). Figure presence does not establish that a current run recreates that exact image.

## Interpretation and reproduction limits

The location of population denominators differs between development variants; parameter sets cannot be moved across them without checking definitions. A complete literature derivation was not reconstructed.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

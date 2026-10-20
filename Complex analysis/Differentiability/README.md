# Complex Domains and Branch Cuts

**Where do chosen complex-function branches exclude points?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../README.md)

---

## Question and approach

The named domain example constructs z²−2 on a complex mesh, filters an argument boundary, and marks its zeros. Other untitled files preserve additional exploratory variants.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [Domain of Differentiablitiy of the given function-1.py](Domain%20of%20Differentiablitiy%20of%20the%20given%20function-1.py) | Top-level experiment; imports: matplotlib.pyplot, numpy |
| [Domain of Differentiablitiy of the given function-2.py](Domain%20of%20Differentiablitiy%20of%20the%20given%20function-2.py) | Top-level experiment; imports: matplotlib.patches, matplotlib.pyplot, numpy |
| [untitled0.py](untitled0.py) | Defined helpers: `rotate_z` |
| [untitled1.py](untitled1.py) | Defined helpers: `rotate_z` |
| [untitled2.py](untitled2.py) | Defined helpers: `rotate_z` |
| [untitled3.py](untitled3.py) | Defined helpers: `rotate_z`, `set_axes_equal` |
| [untitled4.py](untitled4.py) | Defined helpers: `set_axes_equal` |
| [untitled5.py](untitled5.py) | Defined helpers: `set_axes_equal` |

## Inputs, tools, and outputs

Observed import roots: `matplotlib`, `mpl_toolkits`, `numpy`. These observations are not a tested dependency lockfile.

## Interpretation and reproduction limits

A finite mesh and tolerance can miss thin sets. The branch convention and singularities need analytical justification beyond the rendered region.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

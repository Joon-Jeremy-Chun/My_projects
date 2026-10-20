# Behavioral and Awareness Model Variants

**How might behavioral transitions extend basic compartment structures?**

[Portfolio](../../../README.md) · [Evidence & results](../../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../../docs/REPAIRS.md) · [Reproduction](../../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../../docs/CODE_CATALOG.md) · [Collection overview](../../readme.md)

---

## Question and approach

Compare the explicitly named awareness-based and behavioral-transition derivative functions. Read compartment definitions and movement terms before comparing curves.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [(1)Awareness-based SIR model - Isolated susceptible.py](%281%29Awareness-based%20SIR%20model%20-%20Isolated%20susceptible.py) | Defined helpers: `deriv` |
| [(1)Sensitivity Analysis - Behavioral Transition SI-TI model.py](%281%29Sensitivity%20Analysis%20-%20Behavioral%20Transition%20SI-TI%20model.py) | Defined helpers: `deriv`, `sensitivity_analysis` |
| [(1)Three age groups.py](%281%29Three%20age%20groups.py) | Defined helpers: `deriv` |
| [(2)Awareness-Based SIR-TI Model (ABSIR-TI).py](%282%29Awareness-Based%20SIR-TI%20Model%20%28ABSIR-TI%29.py) | Defined helpers: `deriv`, `find_closest_index` |
| [(3)Behavioral Transition SIR-TI Model (BTSIR-TI).py](%283%29Behavioral%20Transition%20SIR-TI%20Model%20%28BTSIR-TI%29.py) | Defined helpers: `deriv`, `find_closest_index` |
| [(9)Behavioral Transition nth wave model.py](%289%29Behavioral%20Transition%20nth%20wave%20model.py) | Defined helpers: `deriv`, `find_closest_index` |

## Inputs, tools, and outputs

Observed import roots: `matplotlib`, `numpy`, `scipy`. These observations are not a tested dependency lockfile.

## Interpretation and reproduction limits

These are exploratory alternatives to the principal thesis sequence. File presence does not establish empirical calibration, publication, or selection as the final model.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../../docs/REPAIRS.md).

---

[Portfolio](../../../README.md) · [Evidence & results](../../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../../docs/REPAIRS.md) · [Reproduction](../../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../../docs/CODE_CATALOG.md)

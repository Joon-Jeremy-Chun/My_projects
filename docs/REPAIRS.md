# October 2026 Completion & Repair Record

[Portfolio](../README.md) · [Evidence](PROJECT_EVIDENCE.md) · [Reproduction](REPRODUCIBILITY.md) · [Catalog](CODE_CATALOG.md)

The owner authorized completing unfinished code, fixing errors, and publishing
the reviewed work. This pass preserves original notebooks/outputs and separates
new reference implementations from earlier research results.

## Four former placeholders completed

| File | Added behavior | Verification |
|---|---|---|
| [Imperfect-vaccine SIS](<../Mathematical Modeling/Modeling in SIS with Vaccination/SIS Model with incomplete vaccine.py>) | Closed S/I/V model with explicit leaky protection, waning, input validation, a solver helper, and a guarded demo. | Conservation, nonnegativity, invalid input, and the perfect-protection limit. |
| [Viscous diffusion](<../Mathematical Modeling/Modeling in water fluid dynamics/ODEs PDEs.py>) | Central spatial differences + BDF for a zero-wall decaying sine mode; analytical reference and guarded demo. | Two-grid convergence to the analytical solution. |
| [Social-cost accounting](<../Mathematical Modeling/Thesis modeling - epidemics and politics of public health/(7.0.1)Model with social costs.py>) | Cost integration on the supplied time grid; vaccination dose-flow accounting; within-window death increments; explicit units contract. | Constant-rate accounting on a nonuniform grid, including pre-existing deaths and a large V stock. |
| [Week 5 visualization](../Mini-projects/MachineLearning2025F/week5_data_visualization.py) | Explicit CSV/output CLI; normalized headers, numeric handling, BMI construction, distributions, and correlations. | Synthetic translated-schema fixture and converter→plotter integration. |

The SIS/diffusion/accounting assumptions are new, explicitly documented reference
choices. They do not invent a recovered historical dataset, a calibrated vaccine
effect, or a revised thesis outcome.

## Repairs to existing implementations

| Area | Before | After |
|---|---|---|
| Perfect-vaccine SIS | Recovery moved I→V; the vaccination parameter moved V→S. | Vaccination moves S→V and SIS recovery moves I→S. |
| Two-group demography | Undefined `mu_A`; constant removal from each adult compartment. | Explicit mortality parameter; proportional adult removal and equal births into susceptible children, conserving total population. |
| Human/vector model | Susceptible-vector loss used an unrelated product of infected-vector terms. | Infection transfers S_m→I_m and mortality removes both vector states; no recruitment is assumed. |
| Translated notebook export | `!ls` caused Python syntax failure and a Colab mount was assumed. | An explicit local `--input` CLI replaces those operations. IQR filtering now accumulates all selected-column filters. |
| Week 1 scripts | `display` existed only implicitly in an interactive environment; hardcoded workstation input. | Explicit optional IPython display with a print fallback; script-relative or environment-selected data directory. |
| Week 5 conversion | Doubled DataSets path and implicit output directory. | Script-relative defaults and explicit input/output CLI; no directory creation merely from import. |
| Classical Gram–Schmidt | Projections used the progressively updated vector. | Classical projections use the original input column; modified GS retains sequential updates. |
| NGM populations | The grouping produced nine groups with overlapping bins, but later code assumed eight/800. | Eight disjoint bands, consistent with companion examples, use all 18 bins exactly once. |
| Main BVP example | RHS sign contradicted the documented `−u''+pu=qx(L−x)` equation. | RHS sign matches that equation. Constant-negative-load animation variants remain explicitly separate. |
| Simpson order study | Undefined function/typo, odd intervals, and cancellation-dominated reference. | Even intervals, validated inputs, sin on [0,π], analytical integral 2, and an inspectable order study. |
| Solar-pond variants | The source vector omitted the scheme's `2*dt` scaling. | Both variants use `2*dt*C(L−x)` with their existing boundary contributions. |
| Hybrid root finder | Incorrect translated midpoint and an insufficiently safeguarded iteration. | Correct midpoint, sign-preserving bracket, guarded secant steps, explicit endpoint/iteration checks. |
| GUE diagnostic | `ks_2samp(ecdf,tcdf)` treated CDF ordinate arrays as samples. | `kstest(spacings,gue_cdf)` uses the analytic CDF of the plotted comparison density. The nominal p-value remains descriptive. |

The code catalog and individual guides link these paths and expose their actual
function names. [SciPy's API reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html)
supports the distinction between two observation samples and CDF arrays.

## Verification performed

```powershell
python tools/validate_repository.py
python -m ruff check --select E9,F821,F822,F823 .
python -m pytest tests/test_repairs.py -q
```

The original critical static audit reported **22 diagnostics in five files**
(20 undefined-name diagnostics and two diagnostics on one syntax line). The
repaired tree passes that audit. **21 regression cases pass** in the inspected
Python 3.11 environment.

The tests in [test_repairs.py](../tests/test_repairs.py) use analytical/invariant
checks rather than declaring success from a script's title or its printed label:

- Closed-population conservation and limiting vaccine cases.
- A decaying sine solution and second-order spatial convergence.
- Correct cost integration on nonuniform observation times.
- Disjoint population aggregation and vector/demographic flow balances.
- QR reconstruction and distinct CGS/MGS roundoff on an ill-conditioned matrix.
- Translated root brackets and fourth-order Simpson behavior.
- GUE CDF support, normalization, and derivative matching the density.
- A manufactured BVP discrete solution and its residual.
- A solar-pond steady profile and mesh-refinement behavior.
- Week 5 schema conversion and real plot files in a temporary directory.

## Preserved research and runtime boundaries

Historical figures, CSVs, and notebook outputs were not rewritten to match new
code. The SPORA economic scripts remain the archived source of their saved
scenario results; their accounting/time-grid assumptions are described in the
research guide. The new accounting helper is the tested flow-based reference.

No full epidemiological calibration/grid search, model training, new crawling,
broker connection, R runtime restoration, or drone flight was performed. Those
are separate operations requiring their actual data, environment, and execution
scope. Passing this suite does not claim that every historical experiment is
fully validated.

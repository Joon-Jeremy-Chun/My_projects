# Data, Environment & Reproduction Guide

[Portfolio](../README.md) · [Evidence](PROJECT_EVIDENCE.md) · [Repairs](REPAIRS.md) · [Catalog](CODE_CATALOG.md)

The archive contains independent experiments with different working-directory,
data, library, and output assumptions. Start with the relevant README; its
reading order is not automatically a verified execution pipeline.

## Reading and validation first

The source catalog and saved figures can be inspected without executing project
scripts. To run the focused maintenance checks in a separate Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-validation.txt
python tools/validate_repository.py
python -m ruff check --select E9,F821,F822,F823 .
python -m pytest tests/test_repairs.py -q
```

`requirements-validation.txt` covers that suite, not every historical project.
The inspected environment uses Python 3.11; historical package compatibility and
a universal version-pinned environment have not been established.

## Collection-specific preparation

| Collection | Inputs and observed tools | Preparation before a new run |
|---|---|---|
| Thesis / SPORA | NumPy, pandas, SciPy, Matplotlib; `Inputs.xlsx`, age-group/state CSVs, fitted matrices. | Match state order, units, dates, and the exact variant. Some parameters are loaded; others are copied into later scripts. Run a small scenario before a full grid. |
| Introductory dynamical systems | NumPy, SciPy, Matplotlib; usually explicit example parameters. | Check normalization, flows, initial conditions, and solver status. |
| Age-PDE / NGM | NumPy, pandas, SciPy; contact/population inputs and existing matrix CSVs. | Check age grouping, interpolation/smoothing, matrix dimensions, and working-directory outputs. An 800×800 calculation can be substantial. |
| Linear algebra / numerical coursework | NumPy, SciPy, Matplotlib; Fraction or SymPy in selected files. | Identify the active matrix, RHS, tolerance, iteration cap, exact reference, and plot destination. |
| Zeta / complex analysis | mpmath, NumPy, SciPy, Matplotlib; prescribed root guesses and series grids. | Check precision, root order/uniqueness, mesh/truncation size, and statistical normalization. |
| Markov/random walks | NumPy, Matplotlib and animation tooling in selected examples. | Inspect seeds, sample counts, step rules, animation backend and file-writing calls. |
| Networks | NetworkX, pandas, Matplotlib; Pillow for some GIF workflows; trade matrix CSV. | Check the meaning/direction of weights and the selected scenario. |
| 1-NN simulation | NumPy, pandas, Matplotlib. | Review N, replications, dimensions, random seed and fixed-target interpretation. |
| ML worksheets | NumPy, pandas, Matplotlib, Seaborn; local CSV/XLS inputs. | Map source columns, distinguish toy labels, and separate preprocessing/model selection/final evaluation. |
| Data API | Requests, pandas, response parsers; public-service endpoint and a deliberately supplied credential. | Verify current endpoint/schema, dates, pagination, and output paths. This task did not call the API. |
| Market Python/notebooks | pandas, NumPy, Matplotlib; yfinance, yahoo_fin, Selenium/BeautifulSoup and Colab references in historical files. | Recover matching data, review network calls and external-service compatibility, and redirect output files in a working copy. |
| R market studies | R; quantmod, dplyr, dygraphs, rvest appear in selected scripts. | Restore a separate R environment and verify data interfaces. R execution was not validated here. |
| Robotics | djitellopy, OpenCV and actual drone/network hardware. | Read as source first. Top-level code can connect, stream, take off, and land; never use these as generic smoke imports. |

This table is based on source observations. Optional imports and branch-specific
dependencies are not a promise that installing the names makes every file run.

## Newly completed references

The following modules have guarded demos and can be imported without their demos
running. Their models and tested scope are described in the repair record.

```powershell
python "Mathematical Modeling/Modeling in SIS with Vaccination/SIS Model with incomplete vaccine.py"
python "Mathematical Modeling/Modeling in water fluid dynamics/ODEs PDEs.py"
python "Mathematical Modeling/Thesis modeling - epidemics and politics of public health/(7.0.1)Model with social costs.py"
```

The first two open illustrative plots when run normally. The cost helper prints
an explicitly synthetic accounting fixture, not a fitted economic outcome.

### Week 5 conversion and visualization

Choose a directory containing the expected Korean 2020 NHIS CSV filenames, then
select an output directory deliberately:

```powershell
python "Mini-projects/MachineLearning2025F/week5_data_sets.py" --input-dir "path/to/Health_2020_Kor" --output-dir "outputs/week5-converted"
python "Mini-projects/MachineLearning2025F/week5_data_visualization.py" --input "outputs/week5-converted/Week5_health_subset.csv" --output-dir "outputs/week5-figures"
```

The converter writes translated CSVs, a merged CSV, and a subset. The plotter
writes two PNGs. The regression test uses synthetic rows in a temporary directory;
it does not inspect private health records or establish clinical validity.

The repaired translated 2017 overview instead takes an explicit
`--input path/to/NHIS_OPEN_GJ_2017_v1.1.csv`. A different year's schema is not a
verified substitute. Week 1 scripts use the project-relative DataSets directory,
or `MY_PROJECTS_DATA_DIR` when deliberately configured.

## Historical output preservation

Run restored experiments in a working copy or explicitly redirected output
directory. Many scripts write `Figures/`, `figures/`, CSV tables, or model files
relative to the working directory. A successful rerun should be recorded with
parameters, inputs, environment, and differences from the earlier artifact.

Do not clear notebook outputs merely to obtain a clean parse. Saved error cells
are historical evidence; JSON parsing is not notebook execution. Likewise, do not
overwrite a published-study figure with a newly corrected reference calculation
and present both as the same experiment.

The ignore rules exclude environments, credentials, caches, and newly generated
IDE/checkpoint files. They do not remove already tracked IDE records,
checkpoints, data, or figures. Data rights and attribution must be considered for
new redistribution; this documentation does not create a dataset license.

## Documentation upkeep

After inspecting a source change, update the hand-authored guide definitions in
`tools/build_documentation.py` and `tools/portfolio_details.py` as appropriate.
Then deliberately run the builder and the static validator. The root README,
evidence, repairs, and reproduction guide are maintained separately.

Actual GitHub rendering and remote publication are separate from a local HTML
preview. Report the state verified, and retain a topic-specific handoff for the
next session.

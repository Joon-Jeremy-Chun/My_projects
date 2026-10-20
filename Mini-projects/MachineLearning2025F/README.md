# Machine Learning Worksheet Sequence

**How do weighted rules lead to search, gradient-based learning, and evaluation?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../README.md)

---

## Question and approach

Read Week 1 linear scoring, Week 2 hidden-node ideas, Week 3 grid/error surfaces, and the gradient-descent handout. Week 5 converts health data and now has an executable descriptive plotting tool; Week 6 compares regression models, while Week 7 explores credit data.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [Data_overview_translated.py](Data_overview_translated.py) | Top-level experiment; imports: argparse, matplotlib.pyplot, numpy, pandas, pathlib, scipy, seaborn, warnings |
| [week1.py](week1.py) | Defined helpers: `to_snake`, `minmax` |
| [week1_basic_algrithem.py](week1_basic_algrithem.py) | Defined helpers: `to_snake`, `minmax` |
| [week2_nonlinear_algrithem.py](week2_nonlinear_algrithem.py) | Defined helpers: `to_snake`, `minmax_fit`, `minmax_transform`, `h1_height`, `h2_weight`, `compute_score` … |
| [week3_Grid_Search_and_Error_Surface with hidden nodes_(Dont run this).py](week3_Grid_Search_and_Error_Surface%20with%20hidden%20nodes_%28Dont%20run%20this%29.py) | Defined helpers: `to_snake`, `load_dataset`, `encode_color`, `fit_minmax`, `transform_minmax`, `act` … |
| [week3_Grid_Search_and_Error_Surface.py](week3_Grid_Search_and_Error_Surface.py) | Defined helpers: `to_snake`, `load_dataset`, `fit_minmax`, `transform_minmax`, `compute_score`, `predict_label` … |
| [week3_Grid_Search_and_Error_Surface_other_for_.py](week3_Grid_Search_and_Error_Surface_other_for_.py) | Defined helpers: `to_snake`, `load_dataset`, `encode_color`, `fit_minmax`, `transform_minmax`, `compute_score` … |
| [week5_data_sets.py](week5_data_sets.py) | Defined helpers: `read_korean_csv`, `convert_and_save_one`, `make_week5_subset`, `main` |
| [week5_data_visualization.py](week5_data_visualization.py) | Defined helpers: `prepare_data`, `make_plots`, `main` |
| [week6_Blood Presure regression.py](week6_Blood%20Presure%20regression.py) | Defined helpers: `report_missing_minmax`, `add_optional_features`, `standardize_fit`, `standardize_apply`, `linear_closed_form_ridge`, `linear_predict` … |
| [week7_CriditCard_Visuals.py](week7_CriditCard_Visuals.py) | Top-level experiment; imports: matplotlib.pyplot, numpy, os, pandas, seaborn |

## Inputs, tools, and outputs

Observed import roots: `IPython`, `argparse`, `glob`, `json`, `math`, `matplotlib`, `numpy`, `os`, `pandas`, `pathlib`, `re`, `scipy`, `seaborn`, `sys`, `time`, `typing`, `warnings`. These observations are not a tested dependency lockfile.

15 preserved tracked visual artifacts are under [CreditCard_Visuals/](CreditCard_Visuals). Figure presence does not establish that a current run recreates that exact image.

## Interpretation and reproduction limits

Data files and absolute historical paths vary. The high-dimensional hidden-node grid explicitly marked Dont run this is not an entry point. New Week 5 functionality is tested with a synthetic fixture.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

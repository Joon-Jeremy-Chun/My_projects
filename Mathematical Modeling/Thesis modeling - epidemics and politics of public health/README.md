# Age-Structured Epidemic Intervention Research

**How can age-specific epidemic dynamics and modeled economic costs be examined together?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../readme.md)

---

## Question and approach

Follow data preparation, transmission-matrix estimation, intervention/vaccination scenarios, and economic comparison. The SPORA-prefixed sequence is a related revision, not a filename-only guarantee of one fully reproducible pipeline.

## Research context and publication

The owner's 2026 résumé records an M.S. in Mathematics at California State University, Los Angeles (2023–2025), with Melisa Hendrata as thesis advisor. The journal's official record independently lists **Joon Chun and Melisa Hendrata** as authors of the related 2026 SIRVD intervention study in **Spora, volume 12, pages 17–28**, DOI **10.61403/2473-5493.1113**. [Journal record](https://ir.library.illinoisstate.edu/spora/vol12/iss1/2/).

The older thesis README names the work *Optimal Intervention Strategies in Age-Structured SIR Models: Epidemiological and Economic Analysis of COVID-19 in Korea*. The journal revision uses an explicit deceased compartment. These related records are distinguished from a claim that every archived variant exactly reproduces the article.

## Study design

The representative pipeline uses three age groups and compartmental ODEs. In the SIRVD variants, each group is ordered **S, I, R, V, D**: susceptible, infectious, recovered, vaccinated, and cumulative deaths. Transmission is represented by a 3×3 matrix; interventions scale its entries or its overall magnitude.

```mermaid
flowchart LR
    A[Historical KDCA series] --> B[Age grouping and constructed states]
    B --> C[Transmission-matrix fitting]
    C --> D[Intervention and vaccination scenarios]
    D --> E[Medical and economic accounting]
    E --> F[Allocation search and saved figures]
```

| Step | Representative source | What to read |
|---|---|---|
| 1 | [Prepare observed series](DataSets_refining.py) | Merge cumulative snapshots, difference within age groups, and aggregate to 0–19, 20–49, and 50+ bands. |
| 2 | [Estimate baseline transmission](%28SPORA%29%281.0%29_BetaMatrix_Fitting.py) | Fit nine nonnegative beta entries with L-BFGS-B against saved I-series targets. |
| 3 | [Examine changing transmission](%28SPORA%29%282.0%29SIRVD_RollingDailyBetaMultiplier_Forecast.py) | Fit daily multiplier matrices and compare reconstructed trajectories and subperiod averages. |
| 4 | [Represent macroeconomic loss](%28SPORA%29%283.0%29GDP_LossFunction_Fitting.py) | Fit an exponential curve to three prescribed multiplier/loss points. |
| 5 | [Compare uniform vaccination](%28SPORA%29%283.1%29Cost-Effectiveness%20of%20Uniform%20Vaccination%20under%20SIRVD.py) | Evaluate the archived 150-day cost and infection-day scenarios. |
| 6 | [Search age allocations](%28SPORA%29%284.0%29SIRVD_FullCostOptimization.py) | For each transmission multiplier, search child/adult rates and infer the senior rate from a capacity constraint. |
| 7 | [Inspect surfaces and profiles](%28SPORA%29%285.0%29Visualiztion_FullCostOptimization.py) | Save per-multiplier summaries, allocation profiles, and cost heatmaps. |

This is a reading sequence. Files differ in their input paths, saved parameters, state dimensions, and output destinations; they are not one verified command-line pipeline.

## Data, target, and fitting details

| Item | Observation in the source |
|---|---|
| Age groups | `0-19`, `20-49`, `50-80+`; the oldest label denotes the aggregated 50+ group. |
| Baseline inputs | `Inputs.xlsx` and `DataSets/Korea_threeGroups_SIRV_parameter.csv`. |
| Baseline fitting | Nonnegative 3×3 beta; mean squared differences of model/target infectious columns; L-BFGS-B. |
| Fitting-window detail | Stage 1 comments describe February; the implemented end comparison includes March 1 (`<=`), unlike its “exclusive” comment. |
| Short-window mortality | Stage 1 sets `mu=0`; stage 2 uses 12 SIRV states and zero death rates despite the broader SIRVD title. |
| Rolling comparison | Stage 2 optimizes each transition using the next day's observed state. This is retrospective fitting, not an untouched out-of-sample forecast. |
| GDP curve | Three prescribed points are fit by `a*exp(b*m)+c`; provenance of the original macroeconomic estimates was not independently reconstructed here. |

<p align="center"><img src="%28SPORA%29figures/Actual_vs_Fitted_Total_I.png" alt="Archived total infectious-series fit" width="850"></p>

*Original saved fit figure. The target series and fitting criterion belong to this archived experiment; a fresh calibration was not run.*

## Saved numerical outcomes

The following values were read from existing CSVs, not regenerated with an optimizer during this work.

| Archived scenario | Value | Source and boundary |
|---|---:|---|
| Lowest stored joint-search cost | **57.6316 billion USD** | `BestScenarioPerBetaM_SIRVD.csv`; stored minimum `57,631,610,585.27886`. |
| Multiplier at that stored minimum | **0.62** | Minimum among 81 saved multiplier rows; not a proven global optimum over all possible policies. |
| Stored age rates at that row | **(0, 0, 0.0370370)** | Child/adult/senior rates under the archived capacity construction. |
| Lowest stored uniform-vaccination cost | **127.8419 billion USD** | `Vaccination_Optimization_ICER_150days_beta1.0.csv`; minimum among 101 saved rows. |
| Uniform rate at that minimum | **0.011** | The archived 150-day, `beta=1` comparison. |

The two rows of optimal cost summarize different scenario families. Their values should not be combined into a new claim about measured economic savings or a current public-health recommendation.

[Open the joint-search CSV](%28SPORA%29figures/BestScenarioPerBetaM_SIRVD.csv) · [Open the uniform-vaccination CSV](%28SPORA%29figures/Vaccination_Optimization_ICER_150days_beta1.0.csv)

<p align="center"><img src="%28SPORA%29figures/OptimalProfiles_vs_m_SIRVD.png" alt="Archived cost and age-specific allocation profiles" width="700"></p>

*Original saved profile figure: transmission multiplier on the horizontal axis, total cost in billion USD, and age-specific vaccination rates. The highlighted point is from the archived scenario search.*

## Economic model: what the code counts

The historical calculation includes medical cost, wage loss, death-related cost, vaccination cost, and GDP loss. The full-cost variant uses explicit cumulative-death increments. It imposes `N1*v1+N2*v2+N3*v3=capacity` when deriving the senior rate.

The original code also charges a per-person vaccination cost repeatedly to the **V stock** and uses population-based allocation capacity while vaccination ODE flows depend on **S**. Its time grid `linspace(0,150,150)` is not unit-spaced, while cost accumulation assumes unit steps. These implementation choices matter for reinterpretation and are retained as historical evidence.

The completed [2026 accounting reference](%287.0.1%29Model%20with%20social%20costs.py) explicitly charges `a*S` dose flows and integrates on the actual time grid. It is tested separately and does not overwrite saved SPORA results or claim a revised article outcome.

## Credits and usage

The coauthored journal record and the résumé advisor information provide the research context. Broader datasets and cited modeling frameworks retain their original source attribution. Detailed individual contribution statements beyond those records remain unverified.

The former thesis README states: “This repository is for academic and research purposes. Feel free to cite, adapt, or extend with attribution.” That historical note is retained here; no new repository-wide software or dataset license is created by this documentation update.


## File-by-file reading guide

<details>
<summary><b>Expand all source files and helper names</b></summary>

| Source | What to inspect |
|---|---|
| [(0.1)Plots-I(t)-NewCases.py](%280.1%29Plots-I%28t%29-NewCases.py) | Top-level experiment; imports: matplotlib.pyplot, pandas |
| [(0.2.1)Plots-I(t)-SIRVmodels.py](%280.2.1%29Plots-I%28t%29-SIRVmodels.py) | Top-level experiment; imports: matplotlib.pyplot, pandas |
| [(0.2.2)Plots-I(t)-14days rolling infections.py](%280.2.2%29Plots-I%28t%29-14days%20rolling%20infections.py) | Top-level experiment; imports: matplotlib.pyplot, os, pandas |
| [(3.1-1)Transmission Matrix.py](%283.1-1%29Transmission%20Matrix.py) | Defined helpers: `load_parameters`, `deriv`, `find_relative_maxima`, `plot_group` |
| [(3.1-2)Transmission Matrix-plot real data-roll over & cumulative.py](%283.1-2%29Transmission%20Matrix-plot%20real%20data-roll%20over%20%26%20cumulative.py) | Defined helpers: `load_parameters`, `deriv`, `method1_new_infected`, `method2_cumulative`, `plot_comparison` |
| [(3.1-3)Create I(t) data sets from Real data (new comformed) with other parameter (ODEs).py](%283.1-3%29Create%20I%28t%29%20data%20sets%20from%20Real%20data%20%28new%20comformed%29%20with%20other%20parameter%20%28ODEs%29.py) | Top-level experiment; imports: matplotlib.pyplot, pandas |
| [(3.1-4)Transmission Matrix-parameter fitting.py](%283.1-4%29Transmission%20Matrix-parameter%20fitting.py) | Defined helpers: `sirv_model`, `objective` |
| [(3.2-1)R_0 computation with Korea  transmission matrix.py](%283.2-1%29R_0%20computation%20with%20Korea%20%20transmission%20matrix.py) | Defined helpers: `power_method` |
| [(3.2-2)R_0 computation.py](%283.2-2%29R_0%20computation.py) | Top-level experiment; imports: sympy |
| [(3.2-3)R_0 computation with reduction.py](%283.2-3%29R_0%20computation%20with%20reduction.py) | Top-level experiment; imports: numpy |
| [(3.3-0)Componetwise - Beta matrix sensetive analysis.py](%283.3-0%29Componetwise%20-%20Beta%20matrix%20sensetive%20analysis.py) | Defined helpers: `load_parameters`, `deriv`, `find_local_maxima`, `simulate_modified_beta`, `plot_component_wise_beta_reduction` |
| [(3.3-1)Componetwise -(best three) Beta matrix sensetive analysis.py](%283.3-1%29Componetwise%20-%28best%20three%29%20Beta%20matrix%20sensetive%20analysis.py) | Defined helpers: `load_parameters`, `deriv`, `simulate_modified_beta`, `plot_component_wise_beta_reduction`, `plot_top3_beta_reductions`, `plot_top3_combined` … |
| [(3.3-1-2)Quarantine at school 0~3 level - graphs.py](%283.3-1-2%29Quarantine%20at%20school%200~3%20level%20-%20graphs.py) | Defined helpers: `load_parameters`, `deriv`, `find_local_maxima`, `compute_peak_reduction_and_delay_all`, `plot_group_infected_4_levels`, `plot_total_infected_4_levels` |
| [(3.3-2-2)Quarantine at workplace 0~3 level - graphs.py](%283.3-2-2%29Quarantine%20at%20workplace%200~3%20level%20-%20graphs.py) | Defined helpers: `load_parameters`, `deriv`, `find_local_maxima`, `compute_peak_reduction_and_delay_all_workplace`, `plot_group_infected_4_levels_workplace`, `plot_total_infected_4_levels_workplace` |
| [(3.3-3-2)Quarantine at school & workplaces 0~3 level - graphs.py](%283.3-3-2%29Quarantine%20at%20school%20%26%20workplaces%200~3%20level%20-%20graphs.py) | Defined helpers: `load_parameters`, `deriv`, `find_local_maxima`, `compute_peak_reduction_and_delay_all_combined`, `plot_group_infected_4_levels_combined`, `plot_total_infected_4_levels_combined` |
| [(3.3-4)Shutdown all the groups 0~3 level + termial level.py](%283.3-4%29Shutdown%20all%20the%20groups%200~3%20level%20%2B%20termial%20level.py) | Defined helpers: `load_parameters`, `deriv`, `find_local_maxima`, `compute_peak_reduction_and_delay_all_population`, `plot_group_infected_4_levels_population`, `plot_total_infected_4_levels_population` |
| [(4.0.0)Code testing - Vaccination.py](%284.0.0%29Code%20testing%20-%20Vaccination.py) | Defined helpers: `vaccination_strategy`, `deriv` |
| [(4.0.1)Vaccination(base code).py](%284.0.1%29Vaccination%28base%20code%29.py) | Defined helpers: `load_parameters`, `vaccination_strategy`, `deriv` |
| [(4.1.1)Vaccination-three period of stages.py](%284.1.1%29Vaccination-three%20period%20of%20stages.py) | Defined helpers: `load_parameters`, `vaccination_strategy`, `deriv` |
| [(4.2.1)Vaccination-diff vaccin rate and three period of stages.py](%284.2.1%29Vaccination-diff%20vaccin%20rate%20and%20three%20period%20of%20stages.py) | Defined helpers: `load_parameters`, `vaccination_strategy`, `deriv` |
| [(4.3.1)Quarantine&Vaccination analysis.py](%284.3.1%29Quarantine%26Vaccination%20analysis.py) | Defined helpers: `load_parameters`, `vaccination_strategy`, `deriv`, `run_simulation` |
| [(4.3.2)Quarantine&Vaccination&Combine analysis.py](%284.3.2%29Quarantine%26Vaccination%26Combine%20analysis.py) | Defined helpers: `load_parameters`, `vaccination_strategy`, `deriv`, `run_simulation` |
| [(4.4.1)Prioritize Vaccination.py](%284.4.1%29Prioritize%20Vaccination.py) | Defined helpers: `load_parameters`, `vaccination_strategy`, `deriv`, `run_example_and_plot`, `plot_total_infections_comparison` |
| [(4.4.2)Prioritize Vaccination keeping tatal capacity.py](%284.4.2%29Prioritize%20Vaccination%20keeping%20tatal%20capacity.py) | Defined helpers: `load_parameters`, `allocate_vaccination_rates`, `get_vaccination_rates`, `vaccination_strategy_factory`, `deriv`, `run_example_and_plot` … |
| [(5.0.1)Korea Case Study Visualiztion.py](%285.0.1%29Korea%20Case%20Study%20Visualiztion.py) | Top-level experiment; imports: matplotlib.pyplot, os, pandas |
| [(5.0.1)test.py](%285.0.1%29test.py) | Defined helpers: `load_parameters`, `deriv`, `plot_total_infected_with_vaccination` |
| [(5.1.0)Beta multipliers_(basic plots).py](%285.1.0%29Beta%20multipliers_%28basic%20plots%29.py) | Defined helpers: `sirv_model` |
| [(5.1.1)Beta multipliers uniformly Korea Case Study2020FEB-APR.py](%285.1.1%29Beta%20multipliers%20uniformly%20Korea%20Case%20Study2020FEB-APR.py) | Defined helpers: `sirv_model` |
| [(5.1.2)Beta uniform multipliers_Korea Case Study.py](%285.1.2%29Beta%20uniform%20multipliers_Korea%20Case%20Study.py) | Defined helpers: `sirv_model`, `simulate_with_multiplier`, `compute_total_peak`, `objective` |
| [(5.1.2-2)Beta uniform multipliers_Korea Case Study.py](%285.1.2-2%29Beta%20uniform%20multipliers_Korea%20Case%20Study.py) | Defined helpers: `sirv_model`, `simulate_with_multiplier`, `compute_total_peak`, `objective` |
| [(5.2.1)Beta multipliers dynamics by day.py](%285.2.1%29Beta%20multipliers%20dynamics%20by%20day.py) | Defined helpers: `sirv_model`, `get_state_from_row`, `objective_day` |
| [(5.2.2)Beta multipliers dynamics level 1,2,3.py](%285.2.2%29Beta%20multipliers%20dynamics%20level%201%2C2%2C3.py) | Defined helpers: `sirv_model`, `get_state_from_row`, `save_fig` |
| [(6.0.1)The first intervention scenario.py](%286.0.1%29The%20first%20intervention%20scenario.py) | Defined helpers: `load_parameters`, `deriv`, `simulate_with_reduction`, `compute_relative_maxima` |
| [(6.0.2)Korea Case Study - 2020 the first intervention scenario.py](%286.0.2%29Korea%20Case%20Study%20-%202020%20the%20first%20intervention%20scenario.py) | Defined helpers: `load_parameters`, `deriv`, `simulate_with_reduction`, `compute_relative_maxima` |
| [(7.0.1)Model with social costs.py](%287.0.1%29Model%20with%20social%20costs.py) | Defined helpers: `compute_social_costs`, `main` |
| [(7.1.1)GDP lost function compuation.py](%287.1.1%29GDP%20lost%20function%20compuation.py) | Defined helpers: `gdp_loss` |
| [(8.2.0)Economic Cost under uniform betaM with time span.py](%288.2.0%29Economic%20Cost%20under%20uniform%20betaM%20with%20time%20span.py) | Defined helpers: `load_parameters`, `deriv`, `run_simulation`, `compute_cost_dataframe` |
| [(9.1.0)Optimization Social Distancing economic benefits(base code) .py](%289.1.0%29Optimization%20Social%20Distancing%20economic%20benefits%28base%20code%29%20.py) | Defined helpers: `deriv`, `run_simulation`, `compute_cost_dataframe` |
| [(9.1.1)Uniformly Vaccinate in economic benefits.py](%289.1.1%29Uniformly%20Vaccinate%20in%20economic%20benefits.py) | Defined helpers: `deriv`, `run_simulation`, `compute_cost_dataframe` |
| [(9.2.1)Optimal_economic benefits in diff age groups_Vaccination.py](%289.2.1%29Optimal_economic%20benefits%20in%20diff%20age%20groups_Vaccination.py) | Defined helpers: `deriv_age_alloc`, `compute_cost_dataframe`, `compute_v3`, `run_simulation_age_alloc` |
| [(9.3.1)Optimal_economic benefits in Level of SD and diff age groups_Vaccination(Heatmaps).py](%289.3.1%29Optimal_economic%20benefits%20in%20Level%20of%20SD%20and%20diff%20age%20groups_Vaccination%28Heatmaps%29.py) | Defined helpers: `deriv_sirv`, `compute_v3`, `compute_cost_dataframe`, `run_simulation_and_cost`, `run_simulation_age_alloc_fixedBM`, `build_cost_array_for_fixedBM` |
| [(9.3.2)Computing I under Optimal_economic benefits in Level of SD and diff age groups_Vaccination.py](%289.3.2%29Computing%20I%20under%20Optimal_economic%20benefits%20in%20Level%20of%20SD%20and%20diff%20age%20groups_Vaccination.py) | Defined helpers: `load_parameters`, `deriv` |
| [(9.3.3)Optimal_economic benefits in Level of SD and diff age groups_Vaccination(Optimal plane).py](%289.3.3%29Optimal_economic%20benefits%20in%20Level%20of%20SD%20and%20diff%20age%20groups_Vaccination%28Optimal%20plane%29.py) | Defined helpers: `deriv_sirv`, `compute_v3`, `compute_cost_dataframe`, `run_simulation_and_cost` |
| [(SPORA)(1.0)_BetaMatrix_Fitting.py](%28SPORA%29%281.0%29_BetaMatrix_Fitting.py) | Defined helpers: `sirvd_model`, `objective` |
| [(SPORA)(2.0)SIRVD_RollingDailyBetaMultiplier_Forecast.py](%28SPORA%29%282.0%29SIRVD_RollingDailyBetaMultiplier_Forecast.py) | Defined helpers: `save_fig`, `sirv_model`, `get_state_from_row` |
| [(SPORA)(3.0)GDP_LossFunction_Fitting.py](%28SPORA%29%283.0%29GDP_LossFunction_Fitting.py) | Defined helpers: `gdp_loss` |
| [(SPORA)(3.1)Cost-Effectiveness of Uniform Vaccination under SIRVD.py](%28SPORA%29%283.1%29Cost-Effectiveness%20of%20Uniform%20Vaccination%20under%20SIRVD.py) | Defined helpers: `save_figure`, `deriv_sirvd`, `run_simulation`, `compute_cost_dataframe` |
| [(SPORA)(4.0)SIRVD_FullCostOptimization.py](%28SPORA%29%284.0%29SIRVD_FullCostOptimization.py) | Defined helpers: `deriv_sirvd`, `compute_v3`, `compute_cost_dataframe`, `run_simulation_and_cost` |
| [(SPORA)(5.0)Visualiztion_FullCostOptimization.py](%28SPORA%29%285.0%29Visualiztion_FullCostOptimization.py) | Defined helpers: `save_figure`, `deriv_sirvd`, `compute_v3`, `compute_cost_dataframe`, `run_simulation_and_cost`, `run_simulation_age_alloc_fixed_m` … |
| [DataSets_refining.py](DataSets_refining.py) | Defined helpers: `map_age_group` |
| [(1)Awareness-based SIR model - Isolated susceptible.py](Other%20models/%281%29Awareness-based%20SIR%20model%20-%20Isolated%20susceptible.py) | Defined helpers: `deriv` |
| [(1)Sensitivity Analysis - Behavioral Transition SI-TI model.py](Other%20models/%281%29Sensitivity%20Analysis%20-%20Behavioral%20Transition%20SI-TI%20model.py) | Defined helpers: `deriv`, `sensitivity_analysis` |
| [(1)Three age groups.py](Other%20models/%281%29Three%20age%20groups.py) | Defined helpers: `deriv` |
| [(2)Awareness-Based SIR-TI Model (ABSIR-TI).py](Other%20models/%282%29Awareness-Based%20SIR-TI%20Model%20%28ABSIR-TI%29.py) | Defined helpers: `deriv`, `find_closest_index` |
| [(3)Behavioral Transition SIR-TI Model (BTSIR-TI).py](Other%20models/%283%29Behavioral%20Transition%20SIR-TI%20Model%20%28BTSIR-TI%29.py) | Defined helpers: `deriv`, `find_closest_index` |
| [(9)Behavioral Transition nth wave model.py](Other%20models/%289%29Behavioral%20Transition%20nth%20wave%20model.py) | Defined helpers: `deriv`, `find_closest_index` |
| [R_0 computation.R](SIRV%20modeling%20in%20R/R_0%20computation.R) | Top-level experiment; imports: none detected |
| [Sensitivity analysis - test.py](Sensitivity%20analysis%20-%20test.py) | Defined helpers: `deriv` |
| [Transmission(contact) matrix in Korea.py](Transmission%28contact%29%20matrix%20in%20Korea.py) | Defined helpers: `get_closest_value` |
| [Transmission(contact) matrix in US.py](Transmission%28contact%29%20matrix%20in%20US.py) | Top-level experiment; imports: numpy |
| [matrix computation.py](matrix%20computation.py) | Top-level experiment; imports: sympy |
| [matrix computation_inverse & eigenvalues.py](matrix%20computation_inverse%20%26%20eigenvalues.py) | Top-level experiment; imports: numpy |
| [matrix computation_inverse & transpose & eigenvalues.py](matrix%20computation_inverse%20%26%20transpose%20%26%20eigenvalues.py) | Top-level experiment; imports: numpy |
| [(4.1.1)Vaccination-three period of stages(test).py](test/%284.1.1%29Vaccination-three%20period%20of%20stages%28test%29.py) | Defined helpers: `load_parameters`, `vaccination_strategy`, `deriv` |

</details>

## Inputs, tools, and outputs

Observed import roots: `SALib`, `cv2`, `datetime`, `matplotlib`, `mpl_toolkits`, `numpy`, `os`, `pandas`, `scipy`, `sympy`, `time`. These observations are not a tested dependency lockfile.

154 preserved tracked visual artifacts are under [Thesis modeling - epidemics and politics of public health/](.), [(SPORA)figures/](%28SPORA%29figures), [DataSets/](DataSets), [Figures/](Figures), [SIRV_Plots/](SIRV_Plots). Figure presence does not establish that a current run recreates that exact image.

## Interpretation and reproduction limits

Preserved outputs reflect the archived assumptions. The new flow-based accounting helper is a separate reference; it does not silently revise the published study.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

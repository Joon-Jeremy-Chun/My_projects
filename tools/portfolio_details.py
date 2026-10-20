"""Hand-authored project narratives used by build_documentation.py."""
THESIS = "Mathematical Modeling/Thesis modeling - epidemics and politics of public health"
NLA = "Numerical Linear Algebra"
TEAM = "Team_Projects/Numerical Methods for ODEs, PDEs and Linear Algebra"
ZETA = "Zeta function and zeros"


def thesis(page, link, image, readmes):
    steps = [
        ("Prepare observed series", "DataSets_refining.py", "Merge cumulative snapshots, difference within age groups, and aggregate to 0–19, 20–49, and 50+ bands."),
        ("Estimate baseline transmission", "(SPORA)(1.0)_BetaMatrix_Fitting.py", "Fit nine nonnegative beta entries with L-BFGS-B against saved I-series targets."),
        ("Examine changing transmission", "(SPORA)(2.0)SIRVD_RollingDailyBetaMultiplier_Forecast.py", "Fit daily multiplier matrices and compare reconstructed trajectories and subperiod averages."),
        ("Represent macroeconomic loss", "(SPORA)(3.0)GDP_LossFunction_Fitting.py", "Fit an exponential curve to three prescribed multiplier/loss points."),
        ("Compare uniform vaccination", "(SPORA)(3.1)Cost-Effectiveness of Uniform Vaccination under SIRVD.py", "Evaluate the archived 150-day cost and infection-day scenarios."),
        ("Search age allocations", "(SPORA)(4.0)SIRVD_FullCostOptimization.py", "For each transmission multiplier, search child/adult rates and infer the senior rate from a capacity constraint."),
        ("Inspect surfaces and profiles", "(SPORA)(5.0)Visualiztion_FullCostOptimization.py", "Save per-multiplier summaries, allocation profiles, and cost heatmaps."),
    ]
    table = "\n".join(f"| {i} | {link(page,THESIS+'/'+name,title)} | {what} |" for i,(title,name,what) in enumerate(steps,1))
    fig = THESIS+"/(SPORA)figures/"
    return f'''## Research context and publication

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
{table}

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

{image(page,fig+'Actual_vs_Fitted_Total_I.png','Archived total infectious-series fit')}

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

{link(page,fig+'BestScenarioPerBetaM_SIRVD.csv','Open the joint-search CSV')} · {link(page,fig+'Vaccination_Optimization_ICER_150days_beta1.0.csv','Open the uniform-vaccination CSV')}

{image(page,fig+'OptimalProfiles_vs_m_SIRVD.png','Archived cost and age-specific allocation profiles', '700')}

*Original saved profile figure: transmission multiplier on the horizontal axis, total cost in billion USD, and age-specific vaccination rates. The highlighted point is from the archived scenario search.*

## Economic model: what the code counts

The historical calculation includes medical cost, wage loss, death-related cost, vaccination cost, and GDP loss. The full-cost variant uses explicit cumulative-death increments. It imposes `N1*v1+N2*v2+N3*v3=capacity` when deriving the senior rate.

The original code also charges a per-person vaccination cost repeatedly to the **V stock** and uses population-based allocation capacity while vaccination ODE flows depend on **S**. Its time grid `linspace(0,150,150)` is not unit-spaced, while cost accumulation assumes unit steps. These implementation choices matter for reinterpretation and are retained as historical evidence.

The completed {link(page,THESIS+'/(7.0.1)Model with social costs.py','2026 accounting reference')} explicitly charges `a*S` dose flows and integrates on the actual time grid. It is tested separately and does not overwrite saved SPORA results or claim a revised article outcome.

## Credits and usage

The coauthored journal record and the résumé advisor information provide the research context. Broader datasets and cited modeling frameworks retain their original source attribution. Detailed individual contribution statements beyond those records remain unverified.

The former thesis README states: “This repository is for academic and research purposes. Feel free to cite, adapt, or extend with attribution.” That historical note is retained here; no new repository-wide software or dataset license is created by this documentation update.
'''


def modeling(page, link, image, readmes):
    return f'''## Representative research path

1. Read the {link(page,readmes[THESIS],'age-structured intervention study')} for the main research question and saved scenario outcomes.
2. Use the foundational epidemic and age-structure guides to understand earlier model variants.
3. Compare the newly completed SIS, diffusion, and flow-accounting references with their explicit assumptions and regression tests.

The principal study connects data preparation, compartment equations, parameter estimation, scenario analysis, and cost accounting. The introductory models provide a separate path into nonlinear dynamics and scientific computing.
'''


def linear_algebra(page, link, image, readmes):
    cg = NLA+'/Conjugate Gradient Method-Exact vs Floating-Point Performance on a 2×2 Linear System'
    return f'''## A useful reading route

| Stage | Mathematical focus | Evidence to inspect |
|---|---|---|
| Direct solves | Elimination, pivots, and LU | Input matrices and row operations. |
| Iterative solves | Jacobi, power iteration, CG, steepest descent | Stopping tests, residuals, iterates, and matrix spectrum. |
| Orthogonalization | Classical and modified Gram–Schmidt | Q/R reconstruction and loss of orthogonality. |
| Overdetermined systems | Least squares, normal equations, SVD pseudoinverse | Rank, residual norm, and solution differences. |
| Error behavior | Heron's method and ill-conditioning | Absolute/relative errors and plotting scales. |

{image(page,cg+'/figures/cg_path_exact.png','Original exact-arithmetic conjugate-gradient path','640')}

*Preserved CG path for the specific 2×2 system. Exact-fraction and float calculations are implemented in the source; the image was not regenerated during documentation work.*
'''


def zeta(page, link, image, readmes):
    return f'''## Three questions, three different numerical summaries

| Experiment | What it computes | Interpretation boundary |
|---|---|---|
| `(1.1.2)` | Roots near prescribed starting guesses; raw consecutive gaps; an exponential KS diagnostic. | Root finding needs order/uniqueness checks; a fitted-mean comparison is not a proof of a spacing law. |
| `(1.2.1)` | Gaps divided by their running average, plotted against a GUE Wigner surmise density. | This running normalization differs from a fully justified unfolding analysis. |
| `(1.3.1)` | First 100 `mp.zetazero` values; normalized gaps and all positive pair distances. | A density-normalized distance histogram is not automatically on the standard pair-correlation scale. |

The repaired `(1.2.1)` diagnostic uses a one-sample KS comparison of observed normalized gaps with the plotted distribution's CDF. The former version passed empirical/theoretical CDF ordinate arrays to `ks_2samp`. SciPy specifies two samples of observations for that function; the API distinction is documented in the {link(page,'docs/REPAIRS.md','repair record')}. [SciPy reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html).

{image(page,ZETA+'/Figures/gue_spacing_comparison.png','Preserved normalized zero-gap histogram and comparison curve')}

*Original stored comparison figure. The source's KS calculation is corrected; this historical image is preserved. No newly estimated p-value is claimed here.*

## Start reading

Begin with the numbered zero-statistics scripts, then inspect the functional-equation and parameterized-function experiments. Keep finite numerical evidence separate from statements about all zeta zeros.
'''


def networks(page, link, image, readmes):
    folder='Minimum_Spanning_Tree_Problem'
    return f'''## From an illustrative road network to a trade graph

The first example specifies six city labels and weighted connections, calls Kruskal's minimum-spanning-tree algorithm, and prints the chosen-edge total. Further variants save intermediate Prim/Kruskal frames and animations.

The trade experiments load a 19×19 matrix labeled as 2022 exports and choose edge cost **1/trade_volume**. A scenario applies a fixed multiplicative reduction to US-connected volumes before recalculating the tree. This parameter is a hypothetical reduction, not a measured tariff effect.

{image(page,folder+'/Figures/kruskal_04.png','Preserved final Kruskal construction frame')}

| Layer | Meaning |
|---|---|
| Road weights | Illustrative construction costs assigned in the code. |
| Trade weights | Reciprocal volumes, chosen to favor large-volume connections. |
| Graph representation | Undirected `networkx.Graph`; reverse-direction insertions can replace an existing edge. |
| Algorithm | An MST minimizes the supplied edge weights; disconnected inputs yield a forest. |

The API behavior is described in the [NetworkX MST reference](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.tree.mst.minimum_spanning_tree.html). Whether a tree is a useful economic summary is a separate modeling judgment.
'''


def statistical_learning(page, link, image, readmes):
    target='Mathematical and Statistical Data Science/1-Nearest Neighber'
    return f'''## Representative experiment

The preserved script studies **1-nearest-neighbor prediction at the origin** under a specified noiseless function. It links a geometric question—how close the nearest sample lies—to prediction error as dimension increases.

{image(page,target+'/figures/panel_D.png','Archived 1-NN MSE, squared bias, and variance by dimension')}

*Saved Monte Carlo plot. The horizontal axis spans dimensions 1–10; the vertical quantities are function-value errors for this simulation. Seed, sample count, and replication count are in the linked study guide.*

The accompanying CSV and geometry panels preserve the experiment's numerical and visual artifacts. They were inspected as historical outputs, not recomputed to establish a new performance result.
'''


def mini_projects(page, link, image, readmes):
    f='Mini-projects/MachineLearning2025F'
    rows=[('1','Week1-Simple Linear Classifier.pdf','Weights, normalization, threshold labels; 10 training IDs and two held-out IDs.'),('2','Week2-Simple Nonlinear Node.pdf','Two hidden nodes and nonlinear score construction.'),('3','Week3-Grid Search_Error Surface.pdf','Search over weights/threshold and inspect an error surface.'),('4','Week4-From Gradient Descent to Machine Learning.pdf','Gradient updates, forward calculation, loss, and backpropagation.'),('6','Week6-Blood Pressure regression.pdf','Preprocessing and regression pipeline for systolic/diastolic targets.'),('7','Week7-Data Visualization & Outlier Detection.pdf','Exploratory plots, correlations, and outlier questions.'),('7 dataset','Week7Dataset-Interpretation.pdf','Interpretation of the preserved credit-data exercise.')]
    table='\n'.join(f'| {week} | {link(page,f+"/"+name,name)} | {description} |' for week,name,description in rows)
    return f'''## Worksheet-to-code map

| Stage | Preserved worksheet | Focus |
|---|---|---|
{table}

Week 5 is represented by conversion/subset code and the newly completed visualization script rather than a separate listed worksheet. The plotting CLI accepts an explicit CSV and output directory, making its side effects visible.

## Evaluation questions worth following

The Week 6 source uses a train/validation split, training-only feature scaling, ridge/linear comparisons, and a NumPy one-hidden-layer MLP with activation/width/learning-rate exploration. A validation-selected model still needs a separate final evaluation before its score is presented as independent test performance.

Some earlier preprocessing is performed before splitting, and toy labels may be derived from input variables. Those design choices should be read in their instructional context. No new best-model score is reported by this documentation update.
'''


def personal(page, link, image, readmes):
    folder='Personal_Projects'
    return f'''## Committed gold-data pipeline

| Stage | Source | Observed behavior |
|---|---|---|
| Fetch | {link(page,folder+'/01_fetch_gold_data.py','01_fetch_gold_data.py')} | yfinance download; default `GC=F`; Date/Price CSV; CLI-configurable dates/output. |
| Features | {link(page,folder+'/02_build_bands_and_plots.py','02_build_bands_and_plots.py')} | Date filtering, full-window normalization, 20-observation moving mean/std, asymmetric upper/lower bands. |
| Visuals | Existing `figures/` | Full-window and zoomed historical band plots. |

{image(page,folder+'/figures/gold_bands_full.png','Archived gold-series asymmetric moving bands')}

*Original committed figure. The original working copy contains a newer local plotting variant and additional mean-reversion drafts; those pre-existing uncommitted files were preserved and are not silently included here.*

Full-window normalization uses information from the chosen analysis interval. Descriptive band plots and historical preference rankings do not establish a deployable or independently tested strategy. No market-data fetch, brokerage connection, or order was made for this task.
'''


def team_projects(page, link, image, readmes):
    return '''## Two complementary paths

| Path | Topics | What connects them |
|---|---|---|
| Numerical-analysis foundations | Root finding, tabulated/function quadrature, least squares | Approximation, error measures, and finite calculations. |
| Differential-equation methods | IVPs, coupled SIR ODEs, BVP/Jacobi, forced diffusion | Translating a continuous problem into an iterative computation. |

The detailed guides identify actual formulas, parameters, and reading order. The repair record explains the corrected Simpson study, safeguarded hybrid solver, BVP forcing sign, and Crank–Nicolson source scaling.
'''


DETAILS = {
    THESIS:thesis,
    'Mathematical Modeling':modeling,
    NLA:linear_algebra,
    ZETA:zeta,
    'Minimum_Spanning_Tree_Problem':networks,
    'Mathematical and Statistical Data Science':statistical_learning,
    'Mini-projects':mini_projects,
    'Personal_Projects':personal,
    'Team_Projects':team_projects,
}

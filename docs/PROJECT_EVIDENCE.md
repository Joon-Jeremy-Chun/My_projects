# Project Evidence & Archived Outcomes

[Portfolio](../README.md) · [Repairs](REPAIRS.md) · [Reproduction](REPRODUCIBILITY.md) · [Code catalog](CODE_CATALOG.md)

Review date: **2026-10-20**. This register keeps source observations, archived
results, new tests, and publication evidence distinct.

| Evidence level | What it establishes |
|---|---|
| Source inspected | An equation, parameter, method, import, input, or output call is present. |
| Saved artifact | A CSV, notebook output, image, or animation preserves a historical result. |
| Focused fresh regression | The stated implementation passes a specific analytical/invariant/fixture check. |
| Author record | The owner's résumé records background or a role; this does not independently reconstruct all work. |
| Official publication record | The journal confirms bibliographic metadata and its own description of the study. |
| Not established / 미확인 | The stronger claim is outside the available evidence or completed verification. |

In maintainer conversation, observations use `[팩트]`, interpretation uses `[추론]`,
and material operational risks use `[주의]`. Public guides explain provenance in
ordinary prose rather than treating every saved value as a reproduced finding.

## Research background

The relevant education/publication sections of **Joon Chun - Resume-F2026.pdf**
were consulted. They record an M.S. in Mathematics at California State University,
Los Angeles, 2023–2025, and Melisa Hendrata as thesis advisor. The original PDF,
contact details, and private workstation paths were not copied into this archive.

The [official Spora article record](https://ir.library.illinoisstate.edu/spora/vol12/iss1/2/)
confirms Joon Chun and Melisa Hendrata as authors, publication year 2026,
volume 12, pages 17–28, and DOI 10.61403/2473-5493.1113. This supports the
publication connection; it does not establish that every archived code variant
is identical to the final article implementation.

The older thesis README and the SPORA-prefixed scripts describe related stages.
The updated [research guide](<../Mathematical Modeling/Thesis modeling - epidemics and politics of public health/README.md>)
preserves their distinction and the earlier usage note.

## Saved thesis scenario results

| Artifact | Observed content | Reporting boundary |
|---|---|---|
| [Joint-search CSV](<../Mathematical Modeling/Thesis modeling - epidemics and politics of public health/(SPORA)figures/BestScenarioPerBetaM_SIRVD.csv>) | 81 rows; minimum stored `MinCost=57631610585.27886` at `beta_m≈0.62`; rates `(0,0,0.03703703650688226)`. | 57.6316 billion USD in that archived scenario grid. No optimization was rerun. |
| [Uniform-vaccination CSV](<../Mathematical Modeling/Thesis modeling - epidemics and politics of public health/(SPORA)figures/Vaccination_Optimization_ICER_150days_beta1.0.csv>) | 101 rows; minimum stored `Total_Cost=127841928465.25807` at `vaccination_rate=0.011`. | 127.8419 billion USD for that 150-day comparison; a different scenario family from the joint search. |
| [Fitted-series figure](<../Mathematical Modeling/Thesis modeling - epidemics and politics of public health/(SPORA)figures/Actual_vs_Fitted_Total_I.png>) | Saved model/target comparison. | Visual fit evidence, not an independent forecast evaluation. |
| [Allocation profiles](<../Mathematical Modeling/Thesis modeling - epidemics and politics of public health/(SPORA)figures/OptimalProfiles_vs_m_SIRVD.png>) | Stored cost and vaccination-rate profiles. | Existing modeled outputs with the archived assumptions, not measured policy savings. |

The public numerical summaries are deliberately tied to these files. They do not
claim optimality over all policies, current empirical validity, or a newly
revalidated economic conclusion.

## Source observations that affect interpretation

| Collection | Verified observation | Interpretation / limit |
|---|---|---|
| Thesis fitting | The baseline uses nonnegative 3×3 beta entries and L-BFGS-B; the implemented end-date test includes March 1. | The “exclusive” comment differs from the code. |
| Thesis rolling stage | Each daily multiplier is optimized using the next observed state. | This is retrospective fitting; it does not establish untouched out-of-sample forecasting. |
| Thesis mortality | Stage 1 has explicit D states with zero mortality; stage 2 uses a 12-state SIRV system with zero mortality. | A filename containing SIRVD does not establish identical state dimensions across scripts. |
| Thesis cost grid | Historical code charges vaccination cost to V stock and accumulates with unit-step assumptions despite `linspace(0,150,150)`. | Reinterpreting or changing the economic conclusions requires a separate analysis; the new flow-accounting reference is kept separate. |
| Gram–Schmidt | The old function labeled classical projected an updated vector. | The repaired source now distinguishes CGS from MGS; the old plot is historical. |
| Zeta statistics | The former two-sample KS call received CDF ordinate arrays. | The corrected diagnostic compares observed gaps to a CDF; dependence/unfolding still constrain inference. |
| Trade network | Reciprocal trade volume is assigned as an edge cost in an undirected graph. | It is a modeling choice; opposite directed entries require an explicit aggregation rule for new economic interpretation. |
| 1-NN | The script specifies a noiseless synthetic response, target origin, seed 42, N=2000, R=300, dimensions 1–10. | Its error curves describe this simulation, not a real-world classifier benchmark. |
| Teaching sequence | Seven PDFs contain worksheets/handouts and synthetic classification examples. | The folder demonstrates instructional material; teaching delivery and individual authorship of every component are not inferred from file presence. |
| Robotics | Flight examples connect and can take off at top level. | No hardware execution was performed; the scripts must not be imported as generic tests. |

Relevant primary API references: [SciPy two-sample KS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html)
and [NetworkX minimum spanning tree](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.tree.mst.minimum_spanning_tree.html).

## Notebook history

| Notebook | Cells | Saved error cells |
|---|---:|---|
| Equity recommendation | 103 | None |
| EPS/PER bands | 48 | 20, 44 |
| EPS base table | 67 | 39, 63 |
| Company EPS/revenue crawl | 16 | 10 |
| Crawl checkpoint | 16 | 10 |

The [catalog](CODE_CATALOG.md) links the exact notebooks. These are saved errors;
they are not fresh failures, and their presence does not establish that the
current third-party service behaves the same way. Notebook outputs were retained.

## Fresh validation and outstanding boundaries

The [repair record](REPAIRS.md) identifies corrected files and **21 passing
regression cases**. The suite checks specified helpers and selected controlled
script configurations. The static validator checks Python parsing, notebook JSON,
Markdown references, and SVG structure.

Not established: complete restoration of historical environments; all-data
end-to-end reproduction; independent epidemiological validation; prospective
investment performance; flight behavior; or exact producer/source correspondence
for every historical image. These boundaries remain explicit rather than being
filled with plausible portfolio claims.

<p align="center"><img src="../assets/readme/modeling.svg" alt="Mathematical Modeling" width="100%"></p>

# Mathematical Modeling

**From compartment models to age-specific intervention and economic scenarios.**

[Portfolio](../README.md) · [Evidence & results](../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../docs/REPAIRS.md) · [Reproduction](../docs/REPRODUCIBILITY.md) · [Full code catalog](../docs/CODE_CATALOG.md)

---

## At a glance

| Item | Description |
|---|---|
| Collection | Mathematical Modeling |
| Subject source inventory | 97 Python · 1 R · 0 notebooks |
| Verification level | Source inspection, selected saved artifacts, and focused regressions described in the repair log. |

## Scope and interpretation

The thesis and its SPORA revision are the representative research sequence. Earlier SIR, SIS, predator–prey, vector, and age-PDE experiments show the development of the modeling ideas. New 2026 reference implementations are identified separately from saved research outputs.

## Representative research path

1. Read the [age-structured intervention study](Thesis%20modeling%20-%20epidemics%20and%20politics%20of%20public%20health/README.md) for the main research question and saved scenario outcomes.
2. Use the foundational epidemic and age-structure guides to understand earlier model variants.
3. Compare the newly completed SIS, diffusion, and flow-accounting references with their explicit assumptions and regression tests.

The principal study connects data preparation, compartment equations, parameter estimation, scenario analysis, and cost accounting. The introductory models provide a separate path into nonlinear dynamics and scientific computing.


## Project and study guides

| Guide | Question |
|---|---|
| [Age-Structured Epidemic Intervention Research](Thesis%20modeling%20-%20epidemics%20and%20politics%20of%20public%20health/README.md) | How can age-specific epidemic dynamics and modeled economic costs be examined together? |
| [Predator–Prey Dynamics](Modeling%20in%202X2%20prey%20and%20predator/README.md) | How do two interacting populations evolve under the chosen nonlinear rates? |
| [SIS Models with Vaccination](Modeling%20in%20SIS%20with%20Vaccination/README.md) | How do vaccination, breakthrough infection, recovery, and waning change a closed population? |
| [Foundational Epidemic ODE Models](Modeling%20in%20epidemics/README.md) | How do compartment and demographic choices change epidemic trajectories? |
| [Discrete Age-Group Epidemic Models](Modeling%20in%20epidemics%20ODEs%20%28discrete%20age%20groups%29/README.md) | How can contact between children, adults, and seniors be represented in ODEs? |
| [Continuous-Age Epidemic Experiments](Modeling%20in%20epidemics%20PDEs%20%28time%2C%20ages%29/README.md) | What changes when age and time are both represented on a grid? |
| [Human–Vector Transmission](Modeling%20in%20human%20and%20vector%20population%28dengue%20fever%29/README.md) | How can human infection and vector infection interact? |
| [Viscous Diffusion: From PDE to ODEs](Modeling%20in%20water%20fluid%20dynamics/README.md) | Can a spatial discretization recover a known velocity-diffusion solution? |
| [Behavioral and Awareness Model Variants](Thesis%20modeling%20-%20epidemics%20and%20politics%20of%20public%20health/Other%20models/README.md) | How might behavioral transitions extend basic compartment structures? |


## Reading and execution

Start with the guide matching your question, inspect its parameters and inputs, then follow links to source. The order is an explanatory reading path; it is not a guarantee that every script executes consecutively.

Read [the reproduction guide](../docs/REPRODUCIBILITY.md) before running code. Source files can load external data, contact services, write artifacts, run large searches, or control hardware. The focused validation suite does not reproduce all historical experiments.

---

[Portfolio](../README.md) · [Evidence & results](../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../docs/REPAIRS.md) · [Reproduction](../docs/REPRODUCIBILITY.md) · [Full code catalog](../docs/CODE_CATALOG.md)

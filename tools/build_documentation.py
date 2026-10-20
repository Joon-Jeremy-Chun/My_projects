"""Build the collection guides and code inventory from inspected source metadata.

Run deliberately from a repository checkout. This writes Markdown and SVG
presentation assets only; it never executes a project script or moves originals.
"""
from pathlib import Path, PurePosixPath
from urllib.parse import quote
import ast
import collections
import html
import json
import os
import subprocess
import textwrap

ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-10-20"
THESIS = "Mathematical Modeling/Thesis modeling - epidemics and politics of public health"
TEAM = "Team_Projects/Numerical Methods for ODEs, PDEs and Linear Algebra"
NLA = "Numerical Linear Algebra"
NGM = "Numerical method for ODEs_PDEs/Age str R_0 compuation-Numerical method for NxN matrix"
ZETA = "Zeta function and zeros"

# slug, public title, short introduction, interpretation/execution boundary.
TOPICS = {
    "Mathematical Modeling": ("modeling", "Mathematical Modeling", "From compartment models to age-specific intervention and economic scenarios.", "The thesis and its SPORA revision are the representative research sequence. Earlier SIR, SIS, predator–prey, vector, and age-PDE experiments show the development of the modeling ideas. New 2026 reference implementations are identified separately from saved research outputs."),
    NLA: ("linear-algebra", "Numerical Linear Algebra", "Linear systems, roundoff, orthogonality, iteration, and least squares.", "Read the chosen matrix and stopping criterion along with each algorithm. Small illustrative systems demonstrate a computation; they are not benchmarks across all matrix classes. The CGS projection correction changes the current experiment, while original figures remain historical."),
    "Numerical method for ODEs_PDEs": ("differential-equations", "Numerical Differential Equations", "Discretized operators, next-generation matrices, and nonlinear phase portraits.", "The 800-group examples use prescribed contact, population, smoothing, and recovery assumptions. The repaired population grouping is consistent with the companion eight-band examples. Old CSVs and figures were retained and have not been regenerated from the corrected code."),
    "Team_Projects": ("team-numerics", "Numerical Analysis Team Projects", "From root finding and quadrature to initial- and boundary-value problems.", "The folder records team-project work; specific collaborator identities and individual task ownership are not established by the reviewed files. This documentation explains the preserved implementations without assigning sole authorship of the entire collection."),
    "Zeta function and zeros": ("zeta", "Zeta Functions & Zero Statistics", "Complex-function experiments, numerical roots, spacings, and comparison curves.", "Finite samples and graphics are exploratory. They do not prove a conjecture. The GUE KS diagnostic now compares spacings to a CDF; dependence and unfolding choices still limit inferential interpretation."),
    "Complex analysis": ("complex", "Complex Analysis", "Series tests, branch cuts, and sequences of functions in computational form.", "Discrete plots illustrate selected examples. A mesh, a partial sum, or numerically located roots cannot establish a general convergence or differentiability theorem. Several zeta-related variants are also represented in the separate zero-statistics collection."),
    "Markov chain Modeling": ("stochastic", "Markov Chains & Random Walks", "Transition sampling, Monte Carlo experiments, and animated paths.", "Some files called Brownian motion implement finite discrete random walks. Random seeds are not uniformly fixed in the historical scripts. Stochastic lightning paths are illustrative and should be distinguished from the electric-potential PDE collection."),
    "Markov chain textbook Qs practices": ("markov-exercises", "Markov Chain Worked Exercises", "Matrix powers, distributions, expectations, and spectral calculations.", "These are small worked exercises associated with the Sheldon M. Ross study folder. Exact textbook edition/problem mapping and independent solution verification are not established for every numbered script."),
    "Minimum_Spanning_Tree_Problem": ("networks", "Minimum Spanning Trees & Networks", "Road-cost examples, graph algorithms, and exploratory trade networks.", "The road costs are illustrative. Trade graphs use reciprocal volume as a chosen edge cost, not a calibrated transport-cost model or a causal estimate of tariffs. Directional trade entries are inserted into an undirected Graph, requiring an explicit aggregation rule before further economic interpretation."),
    "Mathematical and Statistical Data Science": ("statistical-learning", "Mathematical & Statistical Data Science", "Nearest-neighbor geometry, dimensionality, and error decomposition.", "The preserved 1-NN study evaluates a fixed target at the origin with synthetic noiseless responses. It is not a trained real-world classifier or an independent generalization benchmark."),
    "Mini-projects": ("learning", "Machine Learning Mini-Projects", "Worksheets and code connecting hand-built rules to learning and evaluation.", "Seven preserved PDFs explain the instructional sequence. Toy Healthy/Unhealthy labels and favorite-color encodings are teaching devices. Blood-pressure and credit-data experiments are educational and do not establish clinical or lending decision utility."),
    "Data base OpenAPI": ("open-data", "Public Data API Workflows", "Historical Korean COVID-19 collection, grouping, and data correction scripts.", "The code refers to data.go.kr/KDCA interfaces and historical snapshots. Endpoint availability, completeness, response schema, and data redistribution conditions require review before new collection. No current API requests were made for this documentation/repair task."),
    "PDEs and Markov chain Modeling": ("potential", "Electric Potential & PDE Experiments", "Finite differences and boundary scenarios for an illustrative potential field.", "The representative script updates potential values in place, making the sweep Gauss–Seidel-like. A saved field is not evidence of a calibrated physical lightning model or of a coupled stochastic growth algorithm."),
    "Personal_Projects": ("personal", "Personal Market-Data Experiments", "Historical equity notebooks, R probability studies, and a gold-data pipeline.", "This guide covers committed artifacts. Separate uncommitted mean-reversion/streaming extensions remain in the owner's original working folder and are not included in this publication. No profitability or live trading operation is established by these exploratory analyses."),
    "Robotics": ("robotics", "Robotics: Tello Control Examples", "Basic connection, battery checks, takeoff/landing, and camera experiments.", "These scripts have real hardware effects at top level. Do not import or execute them as smoke tests. Static inspection does not establish successful or safe flight; no drone was connected during this work."),
    "Simple Math code": ("algebra", "Small Mathematical Utilities", "Exact fractions, probability arithmetic, and matrix multiplication.", "The committed examples are compact mathematical exercises. The larger model-selection scripts currently present only in the original local working folder are outside this commit and are not represented as published artifacts."),
}

# title, question, actual approach, specific limitation.
GUIDES = {
    THESIS: ("Age-Structured Epidemic Intervention Research", "How can age-specific epidemic dynamics and modeled economic costs be examined together?", "Follow data preparation, transmission-matrix estimation, intervention/vaccination scenarios, and economic comparison. The SPORA-prefixed sequence is a related revision, not a filename-only guarantee of one fully reproducible pipeline.", "Preserved outputs reflect the archived assumptions. The new flow-based accounting helper is a separate reference; it does not silently revise the published study."),
    "Mathematical Modeling/Modeling in 2X2 prey and predator": ("Predator–Prey Dynamics", "How do two interacting populations evolve under the chosen nonlinear rates?", "Read the Lotka–Volterra derivative, example rates and initial conditions, solve_ivp call, and then the phase-portrait variant.", "Parameters are illustrative. No population dataset or biological calibration is supplied for these runs."),
    "Mathematical Modeling/Modeling in SIS with Vaccination": ("SIS Models with Vaccination", "How do vaccination, breakthrough infection, recovery, and waning change a closed population?", "The complete-protection example now sends vaccination S→V and SIS recovery I→S. The completed leaky-vaccine file adds an efficacy parameter and waning, conserving S+I+V.", "The 2026 completion defines explicit reference assumptions; it is not an empirical estimate of vaccine efficacy. Population-conservation and perfect-protection-limit regressions cover the new equations."),
    "Mathematical Modeling/Modeling in epidemics": ("Foundational Epidemic ODE Models", "How do compartment and demographic choices change epidemic trajectories?", "Start with SIR_model.py, compare normalized variants, and inspect the demographic SIR/SEIR and death-compartment experiments. Read each state ordering and incidence scaling independently.", "These are related experiments, not interchangeable implementations of one fitted model. Saved rates and initial populations should not be treated as current epidemiological estimates."),
    "Mathematical Modeling/Modeling in epidemics ODEs (discrete age groups)": ("Discrete Age-Group Epidemic Models", "How can contact between children, adults, and seniors be represented in ODEs?", "Inspect the SIR_i derivative and the 3×3 contact matrix, then compare two-/three-group variants and their 2D/3D views. The representative source credits work by Vishaal Ram and Laura P. Schaposnik.", "The location of population denominators differs between development variants; parameter sets cannot be moved across them without checking definitions. A complete literature derivation was not reconstructed."),
    "Mathematical Modeling/Modeling in epidemics PDEs (time, ages)": ("Continuous-Age Epidemic Experiments", "What changes when age and time are both represented on a grid?", "Read the grid definitions, initial age profile, explicit transport/reaction updates, and age-boundary treatment. The separate two-group example now explicitly balances adult deaths and births.", "Boundary indexing, mesh stability, and positive/conservative solutions require checking per variant. The corrected two-group derivative is covered by a conservation test; the full collection is not thereby PDE-validated."),
    "Mathematical Modeling/Modeling in human and vector population(dengue fever)": ("Human–Vector Transmission", "How can human infection and vector infection interact?", "The four-state example couples S_h/I_h and S_m/I_m. The repaired mosquito infection term transfers susceptible vectors into infected vectors, and the modeled mortality removes both vector classes.", "No vector recruitment is included, so vector population decreases at the chosen mortality rate. This is a qualitative reference, not a dengue forecasting model."),
    "Mathematical Modeling/Modeling in water fluid dynamics": ("Viscous Diffusion: From PDE to ODEs", "Can a spatial discretization recover a known velocity-diffusion solution?", "The former placeholder now solves u_t=νu_xx between stationary walls, starting from sin(πx). Central spatial differences produce an interior ODE system integrated with BDF.", "This October 2026 reference excludes advection and pressure forcing. A two-grid regression checks convergence to the analytical decaying sine mode; it does not validate a general fluid solver."),
    THESIS+"/Other models": ("Behavioral and Awareness Model Variants", "How might behavioral transitions extend basic compartment structures?", "Compare the explicitly named awareness-based and behavioral-transition derivative functions. Read compartment definitions and movement terms before comparing curves.", "These are exploratory alternatives to the principal thesis sequence. File presence does not establish empirical calibration, publication, or selection as the final model."),
    "Complex analysis/Advanced Series Convergence Tests": ("Series Convergence Diagnostics", "What do ratio, root, Raabe, Dirichlet, and Kummer examples look like numerically?", "Read the term-generating functions and finite index range. Compare test sequences, reference lines, and partial sums rather than using the saved plot alone.", "Finite values and labeled limit lines are illustrative observations; asymptotic conclusions require the relevant mathematical hypotheses."),
    "Complex analysis/Differentiability": ("Complex Domains and Branch Cuts", "Where do chosen complex-function branches exclude points?", "The named domain example constructs z²−2 on a complex mesh, filters an argument boundary, and marks its zeros. Other untitled files preserve additional exploratory variants.", "A finite mesh and tolerance can miss thin sets. The branch convention and singularities need analytical justification beyond the rendered region."),
    "Complex analysis/Sequence and Series of Function/Sequence and Series of Function": ("Sequences, Series, and Zeta Experiments", "How can convergence and complex-function behavior be visualized?", "Compare the pointwise/uniformly labeled examples, series truncations, parameterized zeta evaluations, and functional-equation explorations.", "The repeated directory name is preserved. Numerical samples do not prove uniform convergence, analytic continuation, or statements about all zeros."),
    "Markov chain Modeling/Simple examples Markov chaine": ("Simple Markov Chain Examples", "How is a path generated from a state distribution and transition matrix?", "Start with the rainy/sunny two-state example, which samples 100 steps from a specified matrix. Compare Monte Carlo and Poisson/normal-distribution explorations afterward.", "Transition sampling and independently sampled distributions are different mechanisms. Several random experiments are unseeded; exact trajectories vary."),
    "Markov chain Modeling/Brownian Motion Modeling": ("Random Walk and Monte Carlo Visualizations", "How do sampled increments accumulate into paths and endpoint distributions?", "Inspect the allowed steps and probabilities, then the number of steps and replications. One preserved 1D example runs 10,000 replications of 100 ±1 steps.", "The discrete examples are not exact continuous Brownian trajectories. A historical comment referring to 100,000 runs differs from the actual 10,000-run assignment."),
    "Markov chain Modeling/Lightning modeling Modeling": ("Stochastic Lightning-Like Paths", "How can constrained random motion create a branching/path illustration?", "The representative model moves down/down-left/down-right on a finite grid until the bottom is reached. Compare the separate visual variants.", "This is a random-path illustration. It is not coupled to the separate potential-field solver and is not a calibrated discharge simulation."),
    "Markov chain textbook Qs practices/Markov chain -Sheldon M. Ross. Q practices": ("Transition-Matrix Worked Calculations", "How do initial distributions propagate and produce expectations?", "Read the transition-probability and expectation examples before numbered variants. The expectation script computes a distribution after three matrix multiplications and weights it by state values.", "State order and row-/column-vector convention matter. The code is the preserved worked calculation, not an independently checked complete textbook answer key."),
    "Mathematical and Statistical Data Science/1-Nearest Neighber": ("1-NN Geometry and Dimensionality", "How does the nearest observation to the origin behave as dimension grows?", "The script samples in [-1,1]^p and uses f(x)=exp(-||x||²), with seed 42. The Monte Carlo settings are 2,000 observations, 300 replications, and dimensions 1–10. Panels show geometry, distance, MSE, squared bias, and variance.", "The target is the fixed origin and the responses are noiseless. The observed error decomposition is specific to that simulation design."),
    "Mini-projects/MachineLearning2025F": ("Machine Learning Worksheet Sequence", "How do weighted rules lead to search, gradient-based learning, and evaluation?", "Read Week 1 linear scoring, Week 2 hidden-node ideas, Week 3 grid/error surfaces, and the gradient-descent handout. Week 5 converts health data and now has an executable descriptive plotting tool; Week 6 compares regression models, while Week 7 explores credit data.", "Data files and absolute historical paths vary. The high-dimensional hidden-node grid explicitly marked Dont run this is not an entry point. New Week 5 functionality is tested with a synthetic fixture."),
    NLA+"/Conjugate Gradient Method-Exact vs Floating-Point Performance on a 2×2 Linear System": ("Conjugate Gradient: Exact and Floating Arithmetic", "What changes when the same small CG computation uses fractions or float64?", "Read the A=[[4,1],[1,3]], b=[1,-8], x0=[2,1] setup, the CG recurrence, and the rational/float comparison. The companion steepest-descent file supplies a different search direction example.", "The figures depict this 2×2 problem. They do not benchmark convergence or stability over arbitrary systems."),
    NLA+"/Gaussian elimination": ("Gaussian Elimination and LU", "How are linear systems reduced and factored?", "Compare plain elimination, the partial-pivoting variant, and LU construction. Track row operations, pivot handling, and the chosen input system.", "A working small example does not establish robust behavior for singular, nearly singular, or badly scaled matrices."),
    NLA+"/Gram-Schmidt and Modified Gram-Schmidt process": ("Classical and Modified Gram–Schmidt", "How do projection order and finite precision affect an orthogonal basis?", "Both functions return Q and R and display orthogonality measures. The classical function now projects each original input column; the modified function updates the remaining columns.", "The old classical implementation projected an already updated vector. Its stored chart predates the correction. A Hilbert-matrix regression checks reconstruction and distinct roundoff behavior."),
    NLA+"/Heron's method-sqrt(S)": ("Heron's Square-Root Iteration", "How does a square-root iterate approach its target from different initial guesses?", "Read the iterative update, termination parameters, and the regular/semilog/log-log error plots. Compare absolute and relative error definitions.", "Interpret the chosen positive target and arithmetic precision. A log plot of one finite run is not a general numerical proof."),
    NLA+"/Jacobi Method — Iteration Matrix, Spectral Radius, and Convergence": ("Jacobi Iteration and Spectral Radius", "How do the iteration matrix, residual, and actual error relate?", "The script forms D, L, U; builds C=D⁻¹(L+U); computes eigenvalues; and records Jacobi iterates. Inspect the currently active A rather than a commented alternative.", "The stopping test uses successive iterates. Reported residual and error histories should be read separately, and a maximum-iteration exit is not convergence."),
    NLA+"/Least-Squares and Pseudo-Inverse for an Overdetermined System": ("Least Squares and the Pseudoinverse", "How do several constructions solve the same overdetermined system?", "The 5×3 example compares numpy.linalg.lstsq, normal equations, and a manually constructed SVD pseudoinverse, including ranks and residual norms.", "Agreement on this matrix does not establish identical conditioning or numerical reliability of all methods on arbitrary data."),
    NLA+"/Power Method - finding (largest) eigenvalue": ("Power Iteration Examples", "How does repeated matrix multiplication reveal a dominant eigen-direction?", "Compare the 2×2 and 3×3 examples, including normalization, eigenvalue estimate, initial vector, and stopping behavior.", "Convergence depends on the spectrum and initialization. The file title alone is not a guarantee for every supplied matrix."),
    NLA+"/Some projects/Some projects": ("Conditioning and Matrix Geometry", "What can a small numerical experiment show about ill-conditioning?", "Read the active matrix and comparison calculations in ill-condetioned.py alongside its column/scaling figures.", "The original spelling and repeated directory level remain. Physical interpretation or universal error bounds are not inferred from a single figure."),
    NLA+"/nxn system numerical solution": ("Iterative Linear-System Variants", "How is a Jacobi iteration organized for concrete n×n examples?", "Compare the three small-system variants and inspect matrix dimensions, initialization, and the iteration loop.", "These are development examples, not a packaged solver with comprehensive input validation."),
    NGM: ("Next-Generation Matrices and Age Aggregation", "How do contact, population, and recovery assumptions feed a spectral-radius calculation?", "Start with the eight-band population/contact inputs, compare upsampling and smoothing, construct F and V, then inspect NGM=FV⁻¹ and eigenvalue/power/QR variants.", "The repaired NGM script now uses eight disjoint age bands containing all 18 population bins. Historical 800×800 CSVs are preserved and are not certified outputs of the repaired code."),
    "Numerical method for ODEs_PDEs/Parabolic PDE(Diffiusion Equation)": ("Discrete Diffusion Operators", "What does an iteration operator reveal about one diffusion discretization?", "The example constructs small A and B matrices, forms A⁻¹B, and inspects eigenvalues and diagonalization.", "The numerical matrix calculation is distinct from a proof for all meshes and boundary conditions."),
    "Numerical method for ODEs_PDEs/Second order non-linear Equation": ("Nonlinear Second-Order Phase Portraits", "How do parameter choices change the vector field of the chosen system?", "Read the right-hand side, phase-grid parameters, and plotted trajectories. The separate integration exercise is a related numerical exploration.", "A displayed finite vector field is not a full equilibrium/stability classification. Parameter-specific plots retain their original provenance."),
    "Data base OpenAPI/Korea Open Data Portal/Data - Korea Disease Control and Prevention Agency (KDCA)": ("KDCA Historical Data Collection", "How are historical COVID-19 API responses retrieved and organized?", "The simple request specifies one date/region; the multi-date version loops over dates. Other scripts merge frames, correct selected points, and form age-group datasets.", "Requests have external effects. No API credential was tested and no current data were collected. Historical corrections need their provenance checked before using snapshots for new analyses."),
    "PDEs and Markov chain Modeling/Lightening Modeling": ("Potential-Field Boundary Scenarios", "How do boundary values change a grid-based electric potential?", "Inspect the 100×100 grid, cloud/ground boundary values, in-place neighbor averaging, maximum iteration count, and successive-change tolerance. Compare the other two boundary scenarios.", "In-place updates differ from simultaneous Jacobi updates. Max-iteration termination alone does not establish the requested tolerance was reached."),
    "Personal_Projects/stock_market_probability": ("Market Direction and Probability Studies in R", "How are returns, runs, and conditional market summaries explored?", "Inspect quantmod data acquisition, return-sign labeling, lagged run counts, and the selected denominator. Other files explore volatility, P/E bands, dividends, and coin-toss probability examples.", "Overlapping run frequencies are not automatically conditional transition probabilities. Current finance-source compatibility and any predictive trading result are unverified."),
    "Personal_Projects/stock_market_project": ("Historical Equity Recommendation and Earnings Notebooks", "How are market deviations, preferences, EPS, and P/E tables explored?", "Read the recommendation notebook's moving averages, standardized deviations, bounds, rankings, and preference-score functions; then the earnings collection notebooks. The complete catalog records saved error cells.", "External Colab files, browser interfaces, and yahoo_fin services are referenced. Saved outputs and historical errors do not establish current execution or investment performance."),
    "Robotics/Basic Drone contraks/Basic Drone contraks-DJI EDU": ("Tello Battery, Flight, and Camera Examples", "What does the preserved script actually send to the drone?", "The basic script connects, checks battery, and takes off only when the value is at least 20%. The camera example starts a stream, previews, takes off, and lands. Inspect the actual control flow before any hardware use.", "Top-level imports/execution can trigger connection and flight. The existing battery branch is not a complete safety system. No hardware run was performed."),
    "Simple Math code/Algebra": ("Exact Probability and Matrix Arithmetic", "How can short scripts make a mathematical calculation inspectable?", "Fraction Computing.py uses Fraction for total-probability/Bayes arithmetic. Matrix multiply.py preserves a separate linear-algebra calculation.", "The guide covers the committed small examples. Larger local model-selection drafts are intentionally outside the publication scope."),
    TEAM+"/Boundary Value Problem & the Jacobi Method": ("Boundary Values and Jacobi Iteration", "How is a two-endpoint boundary problem converted into a tridiagonal solve?", "The main example uses central differences for −u″+pu=qx(L−x), u(0)=u(L)=0, then iterates with an infinity-norm residual. Its right-hand-side sign now matches that stated equation.", "The animation variants use a constant negative load and therefore represent a different forcing convention. They are preserved as variants, not described as identical copies of the repaired main equation."),
    TEAM+"/Numerical Differentiation and Integration with Big O": ("Differentiation, Quadrature, and Error Order", "How does mesh refinement change an approximation error?", "Compare finite-difference and quadrature formulas against the chosen references. The repaired Simpson study uses even interval counts, sin(x) on [0,π], and the analytical integral 2.", "Subinterval counts and node counts differ. Near machine precision, log-ratios are not stable order estimates; convergence evidence is tied to the actual selected test function."),
    TEAM+"/Numerical Solution for First-order IVP with Big O": ("Initial-Value Problems and Error Studies", "How can a known solution help examine time-stepping accuracy?", "The representative test uses y′=2y/t+t²exp(t), y(1)=0, and y=t²(exp(t)−exp(1)). Inspect its two-step midpoint update and Euler startup before comparing endpoint errors.", "The source contains multiple scenarios and plotting/reference-line choices. A plotted slope alone does not validate every stepper or startup scheme."),
    TEAM+"/Parabolic PDEs - Crank-Nicolson scheme": ("Solar-Pond Diffusion and Crank–Nicolson", "How can a forced diffusion equation be stepped implicitly?", "The two solar-pond scripts form A and B with their scheme scaled by two. Their source vector is now 2Δt·C(L−x), with the existing Dirichlet and zero-flux boundary terms.", "The parameter variants use different grids/time horizons. A manufactured steady-state regression checks spatial convergence after the source fix. No physical solar-pond calibration is claimed."),
    TEAM+"/System of ODEs by numerical method-SIR model": ("SIR Systems and Predictor–Corrector Methods", "How can a coupled SIR model be advanced without a black-box integrator?", "The trapezoidal predictor–corrector example predicts with Euler, corrects with averaged derivatives, and plots the three compartments. Related scripts inspect critical contact values and alternative initial conditions.", "Inspect whether states are proportions or counts before transferring beta values between experiments. This is a method exercise, not an independently fitted epidemic forecast."),
    "Team_Projects/Numerical_analysis/Numerical_interpolation": ("Interpolation, Least Squares, and Numerical Integration", "How do tabulated data and analytic functions lead to numerical estimates?", "The collection compares midpoint, Riemann, trapezoidal, and Simpson variants and includes a least-squares approximation study. Read the particular input definition before selecting a formula.", "Several filenames preserve parallel development variants. A correct reference formula does not establish every data-based branch has been freshly executed."),
    "Team_Projects/Numerical_analysis/Numerical_root_finding": ("Root Finding and a Can-Design Example", "How can bisection and secant steps be combined on a bracket?", "The repaired hybrid function retains a sign-changing interval, safeguards secant steps, uses the true midpoint (a+b)/2, and raises on invalid inputs or iteration exhaustion. Other files preserve separate bisection/secant tables and plots.", "The correction is checked on translated brackets, including a flat cubic near the root. Historical outputs are preserved; the can-design example is not a general certified optimizer."),
}


def put(path, value):
    target = ROOT/path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(line.rstrip() for line in textwrap.dedent(value).strip().splitlines())+"\n", encoding="utf-8")


def url(page, target):
    return quote(os.path.relpath(ROOT/target, (ROOT/page).parent).replace("\\", "/"), safe="/._-")


def link(page, target, label):
    return f"[{label}]({url(page, target)})"


def nav(page):
    return " · ".join(link(page, target, label) for target, label in [("README.md", "Portfolio"), ("docs/PROJECT_EVIDENCE.md", "Evidence & results"), ("docs/REPAIRS.md", "2026 repairs"), ("docs/REPRODUCIBILITY.md", "Reproduction"), ("docs/CODE_CATALOG.md", "Full code catalog")])


def image(page, target, alt, width="850"):
    assert (ROOT/target).is_file(), target
    return f'<p align="center"><img src="{url(page,target)}" alt="{html.escape(alt,quote=True)}" width="{width}"></p>'


def sources():
    tracked = [p for p in subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT, text=True, encoding="utf-8").split("\0") if p]
    result = []
    for path in tracked:
        if not path.lower().endswith((".py", ".r", ".ipynb")):
            continue
        text = (ROOT/path).read_text(encoding="utf-8-sig")
        item = {"path": path, "lines": len(text.splitlines()), "functions": [], "imports": []}
        if path.endswith(".py"):
            tree = ast.parse(text)
            item["functions"] = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            item["imports"] = sorted({n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names})
        elif path.endswith(".ipynb"):
            notebook = json.loads(text)
            item["cells"] = len(notebook["cells"])
            item["errors"] = [i+1 for i,c in enumerate(notebook["cells"]) if any(o.get("output_type")=="error" for o in c.get("outputs", []))]
        else:
            import re
            item["imports"] = sorted(set(re.findall(r"(?:library|require)\s*\(\s*([\w.]+)", text)))
        result.append(item)
    return tracked, result


def role(item):
    if "cells" in item:
        errors = ", ".join(map(str, item["errors"])) or "none"
        return f'{item["cells"]} notebook cells; saved error cells: {errors}.'
    if item["functions"]:
        names = ", ".join("`"+s+"`" for s in item["functions"][:6])
        return "Defined helpers: "+names+(" …" if len(item["functions"])>6 else "")
    return "Top-level experiment; imports: "+(", ".join(item["imports"]) or "none detected")


def file_table(page, items):
    rows = ["| Source | What to inspect |", "|---|---|"]
    for item in items:
        rows.append("| "+link(page,item["path"],PurePosixPath(item["path"]).name)+" | "+role(item).replace("|", "/")+" |")
    return "\n".join(rows)


def cover(slug, title, number):
    color = ["#70e1e8", "#bfa6ff", "#9ad99e", "#ffcc85"][number%4]
    title_lines = textwrap.wrap(title, width=46)
    texts = "".join(f'<text x="48" y="{111+i*46}" font-size="37" font-weight="700" fill="#f8fbff">{html.escape(t)}</text>' for i,t in enumerate(title_lines))
    put(f"assets/readme/{slug}.svg", f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="250" viewBox="0 0 1200 250" role="img" aria-labelledby="title desc">
<title id="title">{html.escape(title)}</title><desc id="desc">Decorative collection cover for a computational mathematics portfolio; not a data plot.</desc>
<rect width="1200" height="250" rx="18" fill="#152335"/><rect width="7" height="250" rx="3" fill="{color}"/>
<g font-family="Arial,sans-serif"><text x="48" y="46" font-size="15" letter-spacing="2" font-weight="700" fill="{color}">{number+1:02d} / COMPUTATIONAL MATHEMATICS</text>{texts}
<text x="48" y="{111+46*len(title_lines)}" font-size="18" fill="#c5d1e1">Research, algorithms, experiments, and the evidence behind them.</text></g>
<path d="M970 210V62M940 176H1160M950 151C990 80 1040 210 1080 105S1140 185 1170 88" fill="none" stroke="{color}" stroke-width="3" opacity="0.14"/></svg>''')


READMES = {folder:folder+("/readme.md" if folder=="Mathematical Modeling" else "/README.md") for folder in TOPICS}
READMES.update({folder:folder+"/README.md" for folder in GUIDES})


def detailed(page, folder):
    from portfolio_details import DETAILS
    return DETAILS[folder](page,link,image,READMES) if folder in DETAILS else ""


def build():
    tracked, items = sources()
    for number,(folder,info) in enumerate(TOPICS.items()):
        slug,title,introduction,boundary = info
        cover(slug,title,number)
        page = READMES[folder]
        entries = [e for e in items if e["path"].startswith(folder+"/")]
        counts = collections.Counter(PurePosixPath(e["path"]).suffix.lower() for e in entries)
        out = image(page,f"assets/readme/{slug}.svg",title,"100%")+f"\n\n# {title}\n\n**{introduction}**\n\n"+nav(page)+"\n\n---\n\n"
        out += "## At a glance\n\n| Item | Description |\n|---|---|\n"
        out += f'| Collection | {title} |\n| Subject source inventory | {counts[".py"]} Python · {counts[".r"]} R · {counts[".ipynb"]} notebooks |\n| Verification level | Source inspection, selected saved artifacts, and focused regressions described in the repair log. |\n\n'
        out += "## Scope and interpretation\n\n"+boundary+"\n\n"+detailed(page,folder)+"\n\n"
        children = [f for f in GUIDES if f.startswith(folder+"/")]
        if children:
            out += "## Project and study guides\n\n| Guide | Question |\n|---|---|\n"
            for child in children:
                out += f'| {link(page,READMES[child],GUIDES[child][0])} | {GUIDES[child][1]} |\n'
        else:
            out += "## Source guide\n\n"+file_table(page, entries)
        out += "\n\n## Reading and execution\n\nStart with the guide matching your question, inspect its parameters and inputs, then follow links to source. The order is an explanatory reading path; it is not a guarantee that every script executes consecutively.\n\n"
        out += "Read "+link(page,"docs/REPRODUCIBILITY.md","the reproduction guide")+" before running code. Source files can load external data, contact services, write artifacts, run large searches, or control hardware. The focused validation suite does not reproduce all historical experiments.\n\n---\n\n"+nav(page)
        put(page,out)
    for folder,info in GUIDES.items():
        page = READMES[folder]
        entries = [e for e in items if e["path"].startswith(folder+"/")]
        title,question,approach,boundary = info
        parent = READMES[folder.split("/")[0]]
        out = f"# {title}\n\n**{question}**\n\n"+nav(page)+" · "+link(page,parent,"Collection overview")+"\n\n---\n\n"
        out += "## Question and approach\n\n"+approach+"\n\n"+detailed(page,folder)+"\n\n"
        out += "## File-by-file reading guide\n\n"
        if len(entries)>15: out += '<details>\n<summary><b>Expand all source files and helper names</b></summary>\n\n'
        out += file_table(page,entries)+"\n\n"
        if len(entries)>15: out += "</details>\n\n"
        imports = sorted({m.split(".")[0] for e in entries for m in e["imports"] if m})
        out += "## Inputs, tools, and outputs\n\nObserved import roots: "+(", ".join("`"+m+"`" for m in imports) or "none detected")+". These observations are not a tested dependency lockfile.\n\n"
        visuals = [p for p in tracked if p.startswith(folder+"/") and p.lower().endswith((".png",".gif"))]
        if visuals:
            dirs = sorted({str(PurePosixPath(p).parent) for p in visuals})
            out += f"{len(visuals)} preserved tracked visual artifacts are under "+", ".join(link(page,d+"/",PurePosixPath(d).name+"/") for d in dirs)+". Figure presence does not establish that a current run recreates that exact image.\n\n"
        out += "## Interpretation and reproduction limits\n\n"+boundary+"\n\n"
        out += "Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in "+link(page,"docs/REPAIRS.md","the repair record")+".\n\n---\n\n"+nav(page)
        put(page,out)
    page = "docs/CODE_CATALOG.md"
    out = "# Complete Source Catalog\n\n"+nav(page)+"\n\n"
    out += "Subject inventory: **287 Python files, 14 R files, and five notebooks**, including one notebook checkpoint; **306 historical source artifacts**. Four previously empty Python files now have explicitly documented 2026 reference implementations. Maintainer tools/tests are listed separately at the end.\n\n"
    out += "The inventory uses tracked subject paths and current source after repair. Import names and helper names are static observations, not proof of complete execution. Notebook error-cell numbers describe saved history. The original working folder's uncommitted extensions are excluded.\n\n"
    for folder,info in TOPICS.items():
        out += "## "+info[1]+"\n\n"+link(page,READMES[folder],"Collection overview")+"\n\n"
        out += "| Source | Lines / cells | Observed helpers or saved history |\n|---|---:|---|\n"
        for item in items:
            if not item["path"].startswith(folder+"/"): continue
            label = item["path"][len(folder)+1:]
            out += "| "+link(page,item["path"],label)+f' | {item.get("cells",item["lines"])} | '+role(item).replace("|","/")+" |\n"
        out += "\n"
    out += "## Maintainer tooling\n\n"+"\n".join("- "+link(page,p,label) for p,label in [("tools/build_documentation.py","Guide/inventory builder"),("tools/portfolio_details.py","Detailed project narratives"),("tools/validate_repository.py","Static structure and Markdown checks"),("tests/test_repairs.py","Focused mathematical regressions")])
    put(page,out)
    print(f"Built {len(READMES)} collection/project guides, {len(TOPICS)} covers, and the source catalog.")


if __name__ == "__main__":
    build()

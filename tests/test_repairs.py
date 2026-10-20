"""Focused mathematical regressions; no network, brokerage, or hardware calls."""
import ast
import contextlib
import io
from pathlib import Path
import runpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.linalg as la
from scipy.integrate import solve_ivp
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Mathematical Modeling/"
TEAM = "Team_Projects/Numerical Methods for ODEs, PDEs and Linear Algebra/"
SIS = MODEL + "Modeling in SIS with Vaccination/"


def functions(path, names, **extra):
    """Load only named helpers from historical scripts with top-level effects."""
    tree = ast.parse((ROOT/path).read_text(encoding="utf-8-sig"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    namespace = {"np": np, "pd": pd, "solve_ivp": solve_ivp, "erf": erf, **extra}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT/path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("efficacy", [0, 0.3, 1])
def test_sis_conserves_population_and_stays_nonnegative(efficacy):
    model = runpy.run_path(str(ROOT/SIS/"SIS Model with incomplete vaccine.py"))
    sol = model["simulate"](efficacy=efficacy)
    assert np.min(sol.y) >= -1e-9
    np.testing.assert_allclose(sol.y.sum(axis=0), 1, atol=1e-9)


def test_perfect_vaccine_limit_matches_sis_recovery_flow():
    leaky = runpy.run_path(str(ROOT/SIS/"SIS Model with incomplete vaccine.py"))["sis_leaky_rhs"]
    perfect = functions(SIS+"SIS Model with Vaccination.py", ["sis_model_vaccine"])["sis_model_vaccine"]
    y = [0.5, 0.2, 0.3]
    np.testing.assert_allclose(leaky(0, y, efficacy=1, waning=0), perfect(y, 0, 0.3, 0.1, 0.05))
    assert leaky(0, y, efficacy=1)[1] < leaky(0, y, efficacy=0)[1]


def test_invalid_sis_configuration_is_rejected():
    model = runpy.run_path(str(ROOT/SIS/"SIS Model with incomplete vaccine.py"))
    with pytest.raises(ValueError):
        model["simulate"](y0=[1, 1, 1])
    with pytest.raises(ValueError):
        model["simulate"](efficacy=1.1)


def test_diffusion_converges_to_analytical_sine_mode():
    model = runpy.run_path(str(ROOT/MODEL/"Modeling in water fluid dynamics/ODEs PDEs.py"))
    errors = []
    for n in [20, 40]:
        x, times, u = model["solve_diffusion"](intervals=n, points=5)
        np.testing.assert_array_equal(u[[0, -1]], 0)
        assert np.min(u) >= -1e-9
        errors.append(np.max(np.abs(u[:, -1]-model["exact_velocity"](x, times[-1]))))
    assert errors[0]/errors[1] > 3.8
    assert errors[1] < 3e-4


def test_social_costs_integrate_flows_on_nonuniform_time_grid():
    model = runpy.run_path(str(ROOT/MODEL/"Thesis modeling - epidemics and politics of public health/(7.0.1)Model with social costs.py"))
    t = np.array([2.0, 2.5, 5.0])
    states = np.tile([100.0, 2.0, 0.0, 900.0, 7.0], (3, 1))
    states[:, 4] = [7, 8, 10]  # Charge 3 new deaths, not the 7 pre-window deaths.
    costs = model["compute_social_costs"](t, states, 0.1, 2, 3, 5, 10, 4)
    last = costs.iloc[-1]
    assert last.Medical_Cost == pytest.approx(12)
    assert last.Wage_Loss == pytest.approx(18)
    assert last.Vaccination_Cost == pytest.approx(150)  # S*a*dose * duration.
    assert last.Death_Cost == pytest.approx(30)
    assert last.GDP_Loss == pytest.approx(12)
    assert last.Total_Cost == pytest.approx(222)
    with pytest.raises(ValueError):
        model["compute_social_costs"]([0, 0, 1], states, 0, 0, 0, 0, 0)


def test_demographic_sir_balances_births_and_deaths():
    fn = functions(MODEL+"Modeling in epidemics PDEs (time, ages)/example.py", ["deriv"])["deriv"]
    result = fn([50, 2, 3, 90, 4, 5], 0, 55, 99, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.01, 0.02)
    assert sum(result) == pytest.approx(0, abs=1e-12)


def test_vector_infection_transfers_population_and_mortality_removes_it():
    fn = functions(MODEL+"Modeling in human and vector population(dengue fever)/Vector-borne diseases model.py", ["si_model"])["si_model"]
    derivative = fn([0.9, 0.1, 0.8, 0.2], 0, 0.1, 0.2, 0.05, 0.1)
    assert sum(derivative[:2]) == pytest.approx(0)
    assert sum(derivative[2:]) == pytest.approx(-0.1)


def test_population_bins_are_disjoint_complete_and_have_eight_groups():
    fn = functions("Numerical method for ODEs_PDEs/Age str R_0 compuation-Numerical method for NxN matrix/NGM_and_R_0.py", ["group_population"])["group_population"]
    grouped = fn(np.arange(1, 19))
    np.testing.assert_array_equal(grouped, [6, 15, 24, 33, 13, 14, 31, 35])
    assert grouped.sum() == sum(range(1, 19))
    assert np.repeat(grouped/100, 100).shape == (800,)


def test_classical_and_modified_gram_schmidt_have_distinct_roundoff_behavior():
    path = "Numerical Linear Algebra/Gram-Schmidt and Modified Gram-Schmidt process/Classical Gram-Schmidt&Modified Gram-Schmidt.py"
    f = functions(path, ["classical_gram_schmidt", "modified_gram_schmidt"])
    V = la.hilbert(8)
    qc, rc = f["classical_gram_schmidt"](V)
    qm, rm = f["modified_gram_schmidt"](V)
    np.testing.assert_allclose(qc@rc, V, atol=1e-12)
    np.testing.assert_allclose(qm@rm, V, atol=1e-12)
    assert np.linalg.norm(qm.T@qm-np.eye(8)) < np.linalg.norm(qc.T@qc-np.eye(8))/10


@pytest.mark.parametrize("a,b,root", [(10, 20, 17.3), (-5, -1, -2.1), (0, 2, 2**0.5)])
def test_safeguarded_hybrid_converges_in_translated_brackets(a, b, root):
    fn = functions("Team_Projects/Numerical_analysis/Numerical_root_finding/hybrid_bisection+secant.py", ["hybrid"])["hybrid"]
    found, count, height = fn(lambda x: (x-root)**3, lambda x: 2*x, a, b, 1e-9, 200)
    assert abs(found-root) <= 1.1e-9
    assert height == 2*found
    assert count <= 200


def test_simpson_is_exact_on_cubics_and_fourth_order_on_sine():
    m = runpy.run_path(str(ROOT/TEAM/"Numerical Differentiation and Integration with Big O/Numerical Integration-Finding Big O.py"))
    assert m["SimC"](lambda x:x**3, 0, 2, 10) == pytest.approx(4)
    _, _, orders = m["convergence_study"](7)
    np.testing.assert_allclose(orders[-3:], 4, atol=0.015)
    with pytest.raises(ValueError):
        m["SimC"](np.sin, 0, np.pi, 3)


def test_gue_cdf_has_correct_support_normalization_and_density():
    m = functions("Zeta function and zeros/(1.2.1)Unfolded Zeta Zero Spacings vs. GUE Wigner Surmise.py", ["p_gue", "gue_cdf"])
    cdf = m["gue_cdf"]
    np.testing.assert_allclose(cdf([-1, 0, 10]), [0, 0, 1], atol=1e-12)
    x = np.linspace(0.1, 4, 100)
    numerical_pdf = (cdf(x+1e-5)-cdf(x-1e-5))/2e-5
    np.testing.assert_allclose(numerical_pdf, m["p_gue"](x), atol=1e-8)


def test_bvp_sign_and_residual_match_the_documented_positive_forcing(monkeypatch):
    path = ROOT/TEAM/"Boundary Value Problem & the Jacobi Method/Boundary Value Problem & the Jacobi Method.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    overrides = {"p": 0, "L": 1, "q": 1, "h": 0.025, "tol": 1e-12, "max_iter": 20000}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in overrides:
            node.value = ast.copy_location(ast.Constant(overrides[node.targets[0].id]), node.value)
    ns = {"__name__": "regression_fixture"}
    monkeypatch.setattr(plt, "show", lambda: None)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(tree, str(path), "exec"), ns)
    x = ns["x"]
    exact = (x-2*x**3+x**4)/12  # -u'' = x(1-x), u(0)=u(1)=0.
    assert np.min(ns["u_approx"]) > 0
    # Central differences of this quartic introduce the known h^2/6 forcing
    # defect; its exact discrete solution differs by h^2*x*(1-x)/12.
    discrete_exact = exact + ns["h"]**2*x*(1-x)/12
    np.testing.assert_allclose(ns["u_complete"], discrete_exact, atol=2e-10)
    assert np.linalg.norm(ns["A"]@ns["u_approx"]-ns["b"], np.inf) <= 1.1e-12
    plt.close("all")


@pytest.mark.parametrize("name", ["Solar Pond - parabolic PDEs - Crank-Nicolson.py", "Solar Pond - parabolic PDEs - Crank-Nicolson (code).py"])
def test_crank_nicolson_source_matches_manufactured_steady_state(name):
    path = ROOT/TEAM/"Parabolic PDEs - Crank-Nicolson scheme"/name
    errors = []
    for nx in [20, 40]:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        overrides = {"Nx": nx, "Nt": 500, "T": 5, "T0": 0}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in overrides:
                node.value = ast.copy_location(ast.Constant(overrides[node.targets[0].id]), node.value)
        ns = {"__name__": "regression_fixture"}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(tree, str(path), "exec"), ns)
        x = np.linspace(0, ns["L"], ns["Nx"]+1)
        exact = ns["C"]*(ns["L"]**2*x/2-ns["L"]*x*x/2+x**3/6)
        errors.append(np.max(np.abs(ns["u"][:, -1]-exact)))
        plt.close("all")
    # The centered ghost-point Neumann boundary has O(h^2) spatial error.
    assert 3.8 < errors[0]/errors[1] < 4.2
    assert errors[1] < 6e-5



def test_week5_visualization_accepts_translated_schema_and_writes_figures(tmp_path):
    m = runpy.run_path(str(ROOT/"Mini-projects/MachineLearning2025F/week5_data_visualization.py"))
    frame = pd.DataFrame({"Height_cm":[160, 170, 180], "Weight_kg":[50, 65, 80], "Systolic_BP":[110, 120, 130]})
    np.testing.assert_allclose(m["prepare_data"](frame).bmi, frame.Weight_kg/(frame.Height_cm/100)**2)
    paths = m["make_plots"](frame, tmp_path)
    assert len(paths) == 2
    assert all(p.is_file() and p.stat().st_size>1000 for p in paths)


def test_week5_converter_and_plotter_work_together_on_a_fixture(tmp_path):
    source = tmp_path/"input"
    source.mkdir()
    frame = pd.DataFrame({"신장(5Cm단위)": [160, 170, 180], "체중(5Kg 단위)": [50, 65, 80],
                          "수축기 혈압": [110, 120, 130], "이완기 혈압": [70, 80, 85]})
    frame.to_csv(source/"National Health Insurance Service_Health Checkup Data_20201231_1.csv", encoding="cp949", index=False)
    model = runpy.run_path(str(ROOT/"Mini-projects/MachineLearning2025F/week5_data_sets.py"))
    output = tmp_path/"converted"
    model["main"](["--input-dir", str(source), "--output-dir", str(output)])
    subset = pd.read_csv(output/"Week5_health_subset.csv")
    assert len(subset) == 3
    np.testing.assert_array_equal(subset.Hypertension, [0, 0, 0])
    plots = runpy.run_path(str(ROOT/"Mini-projects/MachineLearning2025F/week5_data_visualization.py"))["make_plots"](subset, tmp_path/"plots")
    assert all(path.exists() for path in plots)

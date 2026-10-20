#%% Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%% (0) Paths: create ./figures automatically (relative to this .py file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


#%% (1) User parameters (edit here)
SEED = 42

# For A, B panels (visual)
N_AB = 200          # number of sampled points for visual panels
DOMAIN = 1.0        # sample X uniformly from [-DOMAIN, DOMAIN]^p

# For C, D panels (Monte Carlo)
N = 2000            # number of sampled points per replication
R = 300             # replications per dimension
dims = list(range(1, 11))  # dimensions p = 1..10

# Function parameter (ESL-like)
ALPHA = 1.0         # f(x) = exp(-ALPHA * ||x||^2)

# Plot settings
FIGSIZE = (8, 5)


#%% (2) Define the true function f(x)
def f_value(x: np.ndarray, alpha: float = ALPHA):
    """
    True regression function: f(x) = exp(-alpha * ||x||^2)
    x can be shape (p,) or (n,p). Returns scalar or vector.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return float(np.exp(-alpha * np.dot(x, x)))
    return np.exp(-alpha * np.sum(x * x, axis=1))


#%% (3) Helper: sample uniform points in [-DOMAIN, DOMAIN]^p
def sample_uniform(n: int, p: int, domain: float = DOMAIN, rng=None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    return rng.uniform(-domain, domain, size=(n, p))


#%% (4) Helper: 1-NN to the origin (x0 = 0)
def nearest_neighbor_to_origin(X: np.ndarray) -> tuple[int, float]:
    """
    Returns (index, distance) of the nearest neighbor to the origin.
    """
    dists = np.linalg.norm(X, axis=1)
    idx = int(np.argmin(dists))
    return idx, float(dists[idx])


#%% (5) Panel A: 1-NN in one dimension (save to figures/panel_A.png)
def plot_panel_A(rng):
    p = 1
    X = sample_uniform(N_AB, p, rng=rng)[:, 0]
    Y = f_value(X.reshape(-1, 1))

    x0 = 0.0
    idx_nn = int(np.argmin(np.abs(X - x0)))
    x_nn = float(X[idx_nn])
    y_nn = float(Y[idx_nn])
    y0 = f_value(np.array([x0]))

    grid = np.linspace(-DOMAIN, DOMAIN, 600)
    f_grid = f_value(grid.reshape(-1, 1))

    plt.figure(figsize=FIGSIZE)
    plt.plot(grid, f_grid, linewidth=2, label=r"True $f(x)=\exp(-\alpha x^2)$")
    plt.scatter(X, Y, s=18, alpha=0.7, label=r"Samples $(X_i,f(X_i))$")
    plt.scatter([x0], [y0], s=90, marker="o", label=r"Target $(x_0,f(x_0))$")
    plt.scatter([x_nn], [y_nn], s=140, marker="x", linewidths=2,
                label=r"1-NN prediction $f(X_{(1)})$")

    # vertical error line
    plt.plot([x0, x0], [y_nn, y0], linewidth=2)

    plt.title("A: 1-NN in One Dimension")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(FIG_DIR, "panel_A.png"), dpi=300)


#%% (6) Panel B: 1-NN geometry in 1D vs 2D (save to figures/panel_B.png)
def plot_panel_B(rng):
    # 1D sample
    X1 = sample_uniform(N_AB, 1, rng=rng)[:, 0]
    idx1 = int(np.argmin(np.abs(X1 - 0.0)))

    # 2D sample
    X2 = sample_uniform(N_AB, 2, rng=rng)
    idx2, r2 = nearest_neighbor_to_origin(X2)
    x_nn_2d = X2[idx2]

    # Match probability mass between 2D ball and 1D interval
    # total 2D area = (2D)^2 = 4D^2; ball area = pi r2^2
    mass_2d = (np.pi * (r2 ** 2)) / (4.0 * (DOMAIN ** 2))
    # 1D mass for interval [-r1, r1] inside [-D,D] is (2r1)/(2D)=r1/D
    r1 = DOMAIN * mass_2d
    r1 = min(r1, DOMAIN)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: 1D
    ax = axes[0]
    ax.scatter(X1, np.zeros_like(X1), s=18, alpha=0.8)
    ax.scatter([0.0], [0.0], s=90, marker="o", label="Target $x_0=0$")
    ax.scatter([X1[idx1]], [0.0], s=140, marker="x", linewidths=2, label="1-NN in 1D")
    ax.axvspan(-r1, r1, alpha=0.25, label="Mass-matched interval")
    ax.set_xlim(-DOMAIN, DOMAIN)
    ax.set_ylim(-0.3, 0.3)
    ax.set_yticks([])
    ax.set_title("1D (mass-matched region)")
    ax.set_xlabel("x")
    ax.legend(loc="upper right")

    # Right: 2D
    ax = axes[1]
    ax.scatter(X2[:, 0], X2[:, 1], s=18, alpha=0.8, label="Sampled points")
    ax.scatter([0.0], [0.0], s=90, marker="o", label="Target $(0,0)$")
    ax.scatter([x_nn_2d[0]], [x_nn_2d[1]], s=140, marker="x", linewidths=2, label="1-NN in 2D")

    theta = np.linspace(0, 2*np.pi, 500)
    ax.plot(r2*np.cos(theta), r2*np.sin(theta), linewidth=2, label="Radius to 1-NN")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-DOMAIN, DOMAIN)
    ax.set_ylim(-DOMAIN, DOMAIN)
    ax.set_title("2D (radius to nearest neighbor)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")

    fig.suptitle("B: 1-NN Geometry (1D vs 2D)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "panel_B.png"), dpi=300)


#%% (7) Simulation: compute mean distance, MSE, bias^2, variance across dimensions
def run_simulation(rng) -> pd.DataFrame:
    records = []
    f0 = 1.0  # f(0)=1

    for p in dims:
        nn_dists = np.empty(R, dtype=float)
        preds = np.empty(R, dtype=float)

        for r in range(R):
            X = sample_uniform(N, p, rng=rng)
            idx, dist = nearest_neighbor_to_origin(X)
            nn_dists[r] = dist
            preds[r] = f_value(X[idx])  # noiseless: Y=f(X)

        mean_dist = float(np.mean(nn_dists))
        mse = float(np.mean((preds - f0)**2))
        bias2 = float((np.mean(preds) - f0)**2)
        var = float(np.var(preds, ddof=0))

        records.append({
            "dimension_p": p,
            "mean_nn_distance": mean_dist,
            "mse": mse,
            "bias2": bias2,
            "variance": var,
            "mean_pred": float(np.mean(preds))
        })

    return pd.DataFrame.from_records(records)


#%% (8) Panel C: average distance to 1-NN vs dimension (save to figures/panel_C.png)
def plot_panel_C(df: pd.DataFrame):
    plt.figure(figsize=FIGSIZE)
    plt.plot(df["dimension_p"], df["mean_nn_distance"], marker="o", linewidth=2)
    plt.title("C: Average Distance to 1-NN vs Dimension")
    plt.xlabel("Dimension p")
    plt.ylabel("Average distance to 1-NN")
    plt.xticks(df["dimension_p"])
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "panel_C.png"), dpi=300)


#%% (9) Panel D: MSE decomposition vs dimension (save to figures/panel_D.png)
def plot_panel_D(df: pd.DataFrame):
    plt.figure(figsize=FIGSIZE)
    plt.plot(df["dimension_p"], df["mse"], marker="o", linewidth=2, label="MSE")
    plt.plot(df["dimension_p"], df["bias2"], marker="o", linewidth=2, label=r"Bias$^2$")
    plt.plot(df["dimension_p"], df["variance"], marker="o", linewidth=2, label="Variance")
    plt.title("D: MSE, Bias$^2$, Variance vs Dimension (1-NN, noiseless)")
    plt.xlabel("Dimension p")
    plt.ylabel("Value")
    plt.xticks(df["dimension_p"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "panel_D.png"), dpi=300)


#%% (10) Main execution
if __name__ == "__main__":
    rng = np.random.default_rng(SEED)

    # A & B (visual)
    plot_panel_A(rng)
    plot_panel_B(rng)

    # C & D (simulation)
    df = run_simulation(rng)
    print("\nSimulation results:")
    print(df)

    # Save results table too (optional but useful)
    df.to_csv(os.path.join(BASE_DIR, "simulation_results.csv"), index=False)

    plot_panel_C(df)
    plot_panel_D(df)

    plt.show()

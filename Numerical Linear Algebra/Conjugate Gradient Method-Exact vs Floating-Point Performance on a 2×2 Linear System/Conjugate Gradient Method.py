#%% 0. Purpose & Setup
"""
Purpose:
    This script implements the Conjugate Gradient (CG) method for the linear system
        A x = b
    used in Problem 6. The goal is:
        - (b) Write CG code that performs k iterations (user-chosen).
        - (c) Run with exact arithmetic (Fractions).
        - (d) Run again allowing round-off errors (float64).
        - Print all iterates, residuals, and directions.
        - Extra: visualize the CG iteration path in R^2 and save figures.

Notes:
    - CG converges in at most n steps for an n×n SPD matrix.
    - For this 2×2 system, exact CG with Fractions will give the exact solution
      in 2 steps (when started from x0 = (2,1)).
"""

import numpy as np
from fractions import Fraction
import matplotlib.pyplot as plt
import os

# Create folder for saving figures (same pattern as Heron's method script)
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


#%% 1. Conjugate Gradient Method (general k-step version)
def cg_method(A, b, x0, max_iter=2, use_fraction=False):
    """
    Perform max_iter iterations of the Conjugate Gradient (CG) method for Ax = b.

    Parameters
    ----------
    A : 2D array-like
        Symmetric positive definite matrix.
    b : 1D array-like
        Right-hand side.
    x0 : 1D array-like
        Initial guess.
    max_iter : int
        Number of CG iterations to perform.
    use_fraction : bool
        If True, use exact rational arithmetic (Fraction).

    Returns
    -------
    xs : list of iterates x^k
    rs : list of residuals r^k
    ps : list of search directions p^k
    """

    #-------------- Setup arithmetic type --------------
    if use_fraction:
        A = np.array([[Fraction(a) for a in row] for row in A], dtype=object)
        b = np.array([Fraction(bi) for bi in b], dtype=object)
        x = np.array([Fraction(xi) for xi in x0], dtype=object)

        def dot(u, v):
            s = Fraction(0)
            for ui, vi in zip(u, v):
                s += ui * vi
            return s

    else:
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        x = np.array(x0, dtype=float)
        dot = lambda u, v: float(np.dot(u, v))

    #-------------- Initial residual and direction --------------
    r = b - A @ x
    p = r.copy()

    xs = [x.copy()]
    rs = [r.copy()]
    ps = [p.copy()]

    #-------------- Main CG Iterations --------------
    for k in range(max_iter):
        Ap = A @ p
        alpha = dot(r, r) / dot(p, Ap)

        x = x + alpha * p
        r_new = r - alpha * Ap

        xs.append(x.copy())
        rs.append(r_new.copy())

        # Early stop if residual becomes exactly zero
        if use_fraction:
            if all(val == 0 for val in r_new):
                break
        else:
            if np.linalg.norm(r_new) == 0:
                break

        beta = dot(r_new, r_new) / dot(r, r)
        p = r_new + beta * p
        ps.append(p.copy())

        r = r_new

    return xs, rs, ps


#%% 2. Printing Function
def print_cg_results(xs, rs, ps, label="CG run"):
    print(f"\n======= Conjugate Gradient Iteration Results ({label}) =======")
    for k in range(len(xs)):
        print(f"\nx^{k} = {xs[k]}")
        print(f"r^{k} = {rs[k]}")
        if k < len(ps):
            print(f"p^{k} = {ps[k]}")
        print("-----------------------------------------------------")


#%% 3. Part (c) — Exact Arithmetic Run (Fractions)
A = [[4, 1],
     [1, 3]]
b = [1, -8]
x0 = [2, 1]

xs_exact, rs_exact, ps_exact = cg_method(A, b, x0, max_iter=2, use_fraction=True)
print_cg_results(xs_exact, rs_exact, ps_exact, label="Exact (Fraction)")


#%% 4. Part (d) — Floating-Point Round-Off Run (float64)
xs_float, rs_float, ps_float = cg_method(A, b, x0, max_iter=2, use_fraction=False)
print_cg_results(xs_float, rs_float, ps_float, label="Floating-point (float64)")


#%% 5. Extra Visualization: CG Iteration Path in R^2
def plot_cg_path(xs, title, filename):
    """
    Plot the CG iterates in R^2, show on screen, and save to figures/ directory.
    """
    xs = np.array(xs, dtype=float)  # Fractions → floats for plotting

    plt.figure(figsize=(6, 6))
    # plot the points and connecting line
    plt.plot(xs[:, 0], xs[:, 1], 'o-', markersize=8, label="CG iterates")

    # Draw arrows between iterates
    for i in range(len(xs) - 1):
        plt.arrow(xs[i, 0], xs[i, 1],
                  xs[i + 1, 0] - xs[i, 0],
                  xs[i + 1, 1] - xs[i, 1],
                  head_width=0.15, length_includes_head=True,
                  color='red')

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.legend()

    save_path = os.path.join(FIG_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.show()  # Spyder에서 화면에 띄우기


# 실제로 그림 그리기: exact 버전 / floating 버전 둘 다 저장
plot_cg_path(xs_exact, "CG Path in $\\mathbb{R}^2$ (Exact Arithmetic)",
             "cg_path_exact.png")

plot_cg_path(xs_float, "CG Path in $\\mathbb{R}^2$ (Floating-Point)",
             "cg_path_float.png")

# ill_conditioned_q11.py
# Q11: Nearly singular 2x2 system demo (Python/Numpy/Matplotlib)

import numpy as np
import matplotlib.pyplot as plt

def build_system(eps: float):
    A = np.array([[1.0 - eps, 3.0],
                  [3.0,        9.0]], dtype=float)
    b = np.array([1.0, 1.0], dtype=float)
    return A, b

def solve_and_report(eps: float):
    A, b = build_system(eps)
    x = np.linalg.solve(A, b)
    detA = np.linalg.det(A)
    cond2 = np.linalg.cond(A, 2)
    # analytic closed-form for cross-check
    x_analytic = np.array([-(2.0)/(3.0*eps), (2.0+eps)/(9.0*eps)], dtype=float)
    rel_err = np.linalg.norm(x - x_analytic) / np.linalg.norm(x_analytic)

    print("=== Base solve ===")
    print(f"eps = {eps:.3e}")
    print("A =\n", A)
    print("b =", b)
    print("x (numpy)       =", x)
    print("x (analytic)    =", x_analytic)
    print(f"relative error  = {rel_err:.3e}")
    print(f"det(A)          = {detA:.3e}")
    print(f"cond_2(A)       = {cond2:.3e}\n")

    return A, b, x, detA, cond2

def perturbation_demo(A, b, x_base, delta=1e-9):
    """Show sensitivity: tiny change in b leads to huge change in x."""
    b_tilde = b.copy()
    b_tilde[1] += delta  # perturb the 2nd entry slightly
    x_tilde = np.linalg.solve(A, b_tilde)

    rel_b = np.linalg.norm(b_tilde - b) / np.linalg.norm(b)
    rel_x = np.linalg.norm(x_tilde - x_base) / np.linalg.norm(x_base)
    amp = rel_x / rel_b if rel_b != 0 else np.inf

    print("=== Perturbation demo ===")
    print(f"delta on b[1]   = {delta:.3e}")
    print("b_tilde         =", b_tilde)
    print("x_tilde         =", x_tilde)
    print(f"rel. change in b = {rel_b:.3e}")
    print(f"rel. change in x = {rel_x:.3e}")
    print(f"amplification    = {amp:.3e}  (≈ condition number scale)\n")

    return b_tilde, x_tilde, amp

def plot_columns(A, save_path="fig_columns.png"):
    """Geometric intuition: columns nearly collinear."""
    a1 = A[:, 0]
    a2 = A[:, 1]

    plt.figure()
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.quiver(0, 0, a1[0], a1[1], angles='xy', scale_units='xy', scale=1)
    plt.quiver(0, 0, a2[0], a2[1], angles='xy', scale_units='xy', scale=1)
    plt.text(a1[0]*0.55, a1[1]*0.55, "a1", fontsize=10)
    plt.text(a2[0]*0.55, a2[1]*0.55, "a2", fontsize=10)
    plt.title("Columns of A (nearly collinear)")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(-1, 4)
    plt.ylim(-1, 4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    print(f"Saved: {save_path}")

def plot_scaling(eps_list, save_path="fig_scaling.png"):
    """Show ||x(eps)|| versus eps to reveal ~1/eps scaling."""
    norms = []
    for eps in eps_list:
        A, b = build_system(eps)
        x = np.linalg.solve(A, b)
        norms.append(np.linalg.norm(x, 2))

    eps_arr = np.array(eps_list)
    norms = np.array(norms)

    plt.figure()
    plt.loglog(eps_arr, norms, marker='o')
    plt.xlabel("epsilon")
    plt.ylabel("||x(eps)||_2")
    plt.title("Scaling of solution norm vs epsilon (expect ~1/epsilon)")
    plt.grid(True, which="both", linestyle=':')
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    print(f"Saved: {save_path}")

def main():
    # 1) Base solve and diagnostics
    eps = 1e-10
    A, b, x, detA, cond2 = solve_and_report(eps)

    # 2) Perturbation demo (measurement error on b)
    _btilde, _xtilde, _amp = perturbation_demo(A, b, x, delta=1e-9)

    # 3) Geometric intuition: columns of A
    plot_columns(A, save_path="fig_columns.png")

    # 4) Scaling demo across eps grid
    eps_grid = [10.0**(-k) for k in range(1, 11)]  # 1e-1 ... 1e-10
    plot_scaling(eps_grid, save_path="fig_scaling.png")

if __name__ == "__main__":
    main()

#%% 0. Purpose & Setup (Gradient / Steepest Descent Only)
"""
This script implements the (plain) Gradient / Steepest Descent method
for the 2x2 SPD system

    A x = b,   A = [[4, 1],
                    [1, 3]],  b = [1, -8],

starting from x0 = (2, 1).

We run several iterations, print the iterates and residuals, and
visualize the path in R^2 with arrows. This is just for comparison
with the Conjugate Gradient (CG) path; it is not part of the report.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Matrix, RHS, initial guess
A = np.array([[4.0, 1.0],
              [1.0, 3.0]])
b = np.array([1.0, -8.0])
x0 = np.array([2.0, 1.0])

# Figures folder
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


#%% 1. Gradient (Steepest Descent) Method
def gradient_method(A, b, x0, max_iter=10):
    """
    Steepest descent method for solving A x = b with A SPD.

    At each step:
        r_k = b - A x_k
        p_k = r_k
        alpha_k = (r_k^T r_k) / (r_k^T A r_k)
        x_{k+1} = x_k + alpha_k r_k
    """

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    x = np.array(x0, dtype=float)

    xs = [x.copy()]
    rs = []

    r = b - A @ x
    rs.append(r.copy())

    for k in range(max_iter):
        Ar = A @ r
        alpha = np.dot(r, r) / np.dot(r, Ar)

        x = x + alpha * r
        r = b - A @ x

        xs.append(x.copy())
        rs.append(r.copy())

        # optional stopping criterion
        if np.linalg.norm(r) < 1e-12:
            break

    return xs, rs


#%% 2. Run Gradient Method, Print Results, and Plot Path
xs_grad, rs_grad = gradient_method(A, b, x0, max_iter=10)

# Print iterates and residuals
print("======= Gradient (Steepest Descent) Iteration Results =======")
for k, (xk, rk) in enumerate(zip(xs_grad, rs_grad)):
    print(f"\nk = {k}")
    print(f"x^{k} = {xk}")
    print(f"r^{k} = {rk}")
    print(f"||r^{k}||_2 = {np.linalg.norm(rk):.6e}")
    print("-----------------------------------------------------")

# Path visualization in R^2
xs_arr = np.array(xs_grad)

plt.figure(figsize=(6, 6))
plt.plot(xs_arr[:, 0], xs_arr[:, 1], 'o-', markersize=8, label="Gradient iterates")

# Draw arrows between iterates
for i in range(len(xs_arr) - 1):
    plt.arrow(xs_arr[i, 0], xs_arr[i, 1],
              xs_arr[i+1, 0] - xs_arr[i, 0],
              xs_arr[i+1, 1] - xs_arr[i, 1],
              head_width=0.15, length_includes_head=True,
              color='red')

plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("Gradient (Steepest Descent) Path in $\\mathbb{R}^2$")
plt.grid(True)
plt.axis('equal')
plt.legend()

save_path = os.path.join(FIG_DIR, "grad_path.png")
plt.savefig(save_path, dpi=300)
plt.show()

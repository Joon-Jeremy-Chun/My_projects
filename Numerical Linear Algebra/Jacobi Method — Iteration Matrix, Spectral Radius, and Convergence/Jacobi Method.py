#%% Purpose
# Problem 2: Convergence Analysis of the Jacobi Method.
# This script sets up the linear system, constructs the Jacobi iteration matrix,
# checks diagonal dominance, and computes eigenvalues and the spectral radius.

#%% 0. Setup: import libraries and define matrices
import numpy as np
import matplotlib.pyplot as plt
import os

# Create folder for saving figures (shared with other problems)
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Coefficient matrix A and right-hand side vector b
# A = np.array([[5.0, -1.0,  2.0],
#               [3.0,  8.0, -2.0],
#               [1.0,  1.0,  4.0]])

A = np.array([[5.0, -1.0,  2.0],
              [3.0,  8.0, -2.0],
              [1.0,  1.0,  2.0]])

b = np.array([12.0, -25.0, 6.0])


def is_strictly_diagonally_dominant(M):
    """
    Check if a square matrix M is strictly diagonally dominant by rows.

    Returns
    -------
    bool
        True if |a_ii| > sum_{j!=i} |a_ij| for all rows i.
    """
    n = M.shape[0]
    for i in range(n):
        diag = abs(M[i, i])
        off_diag_sum = np.sum(np.abs(M[i, :])) - diag
        if not (diag > off_diag_sum):
            return False
    return True
#%% 1. (a)–(c): diagonal dominance, Jacobi iteration matrix C, eigenvalues and spectral radius
print("Coefficient matrix A:")
print(A)
print("\nRight-hand side b:")
print(b)

# (a) Check strict diagonal dominance
is_dd = is_strictly_diagonally_dominant(A)
print("\n(a) Is A strictly diagonally dominant by rows?")
print(f"Answer: {is_dd}")

# Decompose A = D - L - U  (here we store L and U as usual lower/upper parts)
D = np.diag(np.diag(A))
L = -np.tril(A, -1)   # negative strictly lower part
U = -np.triu(A,  1)   # negative strictly upper part

# (b) Jacobi iteration matrix C = D^{-1}(L + U) and vector d = D^{-1} b
D_inv = np.linalg.inv(D)
C = D_inv @ (L + U)
d = D_inv @ b

print("\n(b) Jacobi iteration matrix C and vector d:")
print("C =")
print(C)
print("\nd =")
print(d)

# (c) Eigenvalues of C and its spectral radius
eigvals, _ = np.linalg.eig(C)
rho_C = max(abs(eigvals))

print("\n(c) Eigenvalues of C:")
for idx, lam in enumerate(eigvals):
    print(f"lambda_{idx+1} = {lam:.10f}")

print(f"\nSpectral radius rho(C) = {rho_C:.10f}")

#%% 2. (d) Jacobi iteration: implementation and one test run
def jacobi(A, b, x0, max_iter=50, tol=1e-8):
    """
    Perform the Jacobi iterative method for solving Ax = b.

    Parameters
    ----------
    A : numpy.ndarray
        Coefficient matrix (n x n).
    b : numpy.ndarray
        Right-hand side vector (n,).
    x0 : numpy.ndarray
        Initial guess for the solution (n,).
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for the stopping criterion based on consecutive iterates.

    Returns
    -------
    X_hist : numpy.ndarray
        Array of shape (k+1, n) containing all iterates x^(0), x^(1), ..., x^(k),
        where k is the last iteration index.
    """
    A = A.astype(float)
    b = b.astype(float)
    x = x0.astype(float).copy()

    n = A.shape[0]
    X_hist = np.zeros((max_iter + 1, n))
    X_hist[0, :] = x

    for k in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            sigma = 0.0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x[j]
            x_new[i] = (b[i] - sigma) / A[i, i]

        X_hist[k + 1, :] = x_new

        # Stopping criterion based on the change between iterates
        if np.linalg.norm(x_new - x, ord=2) < tol:
            X_hist = X_hist[: k + 2, :]  # truncate unused rows
            print(f"Jacobi method converged in {k+1} iterations.")
            return X_hist

        x = x_new

    print(f"Jacobi method reached the maximum number of iterations ({max_iter}).")
    return X_hist

# Choose an initial guess and run Jacobi's method
x0 = np.zeros_like(b)  # for example: start from the zero vector
max_iter = 50
tol = 1e-8

jacobi_history = jacobi(A, b, x0, max_iter=max_iter, tol=tol)

print("\nFinal Jacobi iterate (last row of jacobi_history):")
print(jacobi_history[-1, :])

#%% 3. (e) Plot iterative solution components vs iteration
# Extract iteration indices and components from Jacobi history
k_vals = np.arange(jacobi_history.shape[0])
x1_vals = jacobi_history[:, 0]
x2_vals = jacobi_history[:, 1]
x3_vals = jacobi_history[:, 2]

plt.figure(figsize=(6, 4))
plt.plot(k_vals, x1_vals, marker='o', label=r"$x_1^{(k)}$")
plt.plot(k_vals, x2_vals, marker='s', label=r"$x_2^{(k)}$")
plt.plot(k_vals, x3_vals, marker='^', label=r"$x_3^{(k)}$")

plt.xlabel("Iteration k")
plt.ylabel("Component values")
plt.title("Jacobi Iterates for Each Component")
plt.grid(True)
plt.legend()

# Save figure to the figures/ directory
plt.savefig(os.path.join(FIG_DIR, "jacobi_iterates_components.png"), dpi=300)
plt.show()
#%% 4. (f) Compute the exact solution x_star
x_star = np.linalg.solve(A, b)

print("\n(f) Exact solution x_star (A^{-1} b):")
print(x_star)
#%% 5. (g) Semilogy plot of the 2-norm of the error
# Compute error vectors and their 2-norms
errors = jacobi_history - x_star  # each row: x^{(k)} - x_star
error_norms = np.linalg.norm(errors, axis=1)  # 2-norm for each k
k_vals = np.arange(jacobi_history.shape[0])

print("\n(g) 2-norm of the error ||x^(k) - x_star||_2:")
for k, val in zip(k_vals, error_norms):
    print(f"k = {k:2d},  ||error||_2 = {val:.6e}")

# Semilogy plot of the error norm
plt.figure(figsize=(6, 4))
plt.semilogy(k_vals, error_norms, marker='o')
plt.xlabel("Iteration k")
plt.ylabel(r"$\|x^{(k)} - x_\ast\|_2$")
plt.title("Jacobi Method: Error Norm (2-norm)")
plt.grid(True, which="both")

plt.savefig(os.path.join(FIG_DIR, "jacobi_error_norm_semilogy.png"), dpi=300)
plt.show()
#%% 6. (h) Semilogy plot of the 2-norm of the residual
# Compute residuals r^{(k)} = b - A x^{(k)}
residuals = np.zeros_like(jacobi_history)
for idx, xk in enumerate(jacobi_history):
    residuals[idx, :] = b - A @ xk

residual_norms = np.linalg.norm(residuals, axis=1)

print("\n(h) 2-norm of the residual ||b - A x^(k)||_2:")
for k, val in zip(k_vals, residual_norms):
    print(f"k = {k:2d},  ||residual||_2 = {val:.6e}")

# Semilogy plot of the residual norm
plt.figure(figsize=(6, 4))
plt.semilogy(k_vals, residual_norms, marker='o')
plt.xlabel("Iteration k")
plt.ylabel(r"$\|b - A x^{(k)}\|_2$")
plt.title("Jacobi Method: Residual Norm (2-norm)")
plt.grid(True, which="both")

plt.savefig(os.path.join(FIG_DIR, "jacobi_residual_norm_semilogy.png"), dpi=300)
plt.show()

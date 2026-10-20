#%% Purpose
# Problem 4: Least-squares solution of an overdetermined system.
# We consider a 5x3 linear system Ax = b with no exact solution.
# (b) Compute the least-squares solution using np.linalg.lstsq
#     (Python analogue of MATLAB's A\b).
# (c) Compute the same solution using the normal equations A^T A x = A^T b
#     and compare the two results.

#%% 0. Setup: import libraries, create figure directory, define A and b
import numpy as np
import matplotlib.pyplot as plt
import os

# Create folder for saving figures (shared with other problems)
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Nicer printing
np.set_printoptions(precision=6, suppress=True)

# Coefficient matrix A (5x3) and right-hand side vector b (5x1)
A = np.array([[ 5.0, -1.0,  2.0],
              [ 3.0,  8.0, -2.0],
              [ 1.0,  1.0,  4.0],
              [-1.0,  1.0,  1.0],
              [ 1.0, 10.0, -2.0]])

b = np.array([12.0, -25.0, 6.0, 1.0, 10.0])

print("=== Problem 4: Least-Squares System Ax = b ===\n")
print("Matrix A (5x3):")
print(A)
print("\nVector b (5x1):")
print(b)

# Rank check (matches part (a) discussion)
rank_A = np.linalg.matrix_rank(A)
print(f"\nRank(A) = {rank_A} (full column rank = 3)\n")

#%% (b) Least-squares solution via np.linalg.lstsq (analogue of MATLAB A\\b)
print("=== (b) Least-squares solution using np.linalg.lstsq (MATLAB A\\\\b analogue) ===")

# lstsq returns: x, residuals, rank, singular values
x_ls, residuals, rank_lstsq, svals = np.linalg.lstsq(A, b, rcond=None)

print("\nLeast-squares solution x_ls:")
print(x_ls)

# Compute Ax_ls and residual r = b - Ax_ls
b_hat_ls = A @ x_ls
residual_ls = b - b_hat_ls
res_norm_ls = np.linalg.norm(residual_ls, ord=2)

print("\nPredicted b_hat = A x_ls:")
print(b_hat_ls)

print("\nResidual r_ls = b - A x_ls:")
print(residual_ls)
print(f"\n2-norm of residual ||r_ls||_2 = {res_norm_ls:.6e}")

print(f"\nRank reported by lstsq = {rank_lstsq}")
print("Singular values of A:")
print(svals)
print("\n(End of part (b))\n")

#%% (c) Least-squares solution via Normal Equations A^T A x = A^T b
print("=== (c) Least-squares solution using the Normal Equations A^T A x = A^T b ===")

# Form normal-equation matrices
AtA = A.T @ A   # 3x3
Atb = A.T @ b   # 3x1

print("\nMatrix A^T A:")
print(AtA)

print("\nVector A^T b:")
print(Atb)

# Solve (A^T A) x = A^T b
x_normal = np.linalg.solve(AtA, Atb)

print("\nNormal-equation solution x_normal:")
print(x_normal)

# Compute Ax_normal and residual
b_hat_normal = A @ x_normal
residual_normal = b - b_hat_normal
res_norm_normal = np.linalg.norm(residual_normal, ord=2)

print("\nPredicted b_hat_normal = A x_normal:")
print(b_hat_normal)

print("\nResidual r_normal = b - A x_normal:")
print(residual_normal)
print(f"\n2-norm of residual ||r_normal||_2 = {res_norm_normal:.6e}")

# Compare the two solutions
diff_x = np.linalg.norm(x_normal - x_ls, ord=2)
print("\n=== Comparison between lstsq solution and normal-equation solution ===")
print("x_ls       (from np.linalg.lstsq):")
print(x_ls)
print("\nx_normal  (from normal equations):")
print(x_normal)
print(f"\n2-norm of difference ||x_ls - x_normal||_2 = {diff_x:.6e}")
print("\n(They should agree up to numerical roundoff.)\n")
#%% (d) Least-squares solution using SVD-based pseudo-inverse (manual construction)
print("=== (d) Least-squares solution using SVD-based pseudo-inverse (manual) ===")

# Compute the (reduced) SVD of A
# A = U Σ V^T, with U (5x3), Σ (3x3), V^T (3x3) because full_matrices=False
U, s, Vt = np.linalg.svd(A, full_matrices=False)

print("\nLeft singular vectors U (5x3):")
print(U)
print("\nSingular values s:")
print(s)
print("\nRight singular vectors V^T (3x3):")
print(Vt)

# Build Σ^{-1} manually using a for-loop
Sigma_inv = np.zeros((len(s), len(s)))  # 3x3
tol = 1e-14
for i in range(len(s)):
    if s[i] > tol:
        Sigma_inv[i, i] = 1.0 / s[i]
    else:
        Sigma_inv[i, i] = 0.0  # in case of tiny singular values

print("\nDiagonal matrix Σ^{-1}:")
print(Sigma_inv)

# Recall: for reduced SVD, A^+ = V Σ^{-1} U^T
V = Vt.T
A_pinv_svd = V @ Sigma_inv @ U.T

print("\nSVD-based pseudo-inverse A^+ (manual):")
print(A_pinv_svd)

# Compute the least-squares solution x = A^+ b
x_svd = A_pinv_svd @ b

print("\nSVD-based pseudo-inverse solution x_svd:")
print(x_svd)

# Predicted b and residual
b_hat_svd = A @ x_svd
residual_svd = b - b_hat_svd
res_norm_svd = np.linalg.norm(residual_svd, ord=2)

print("\nPredicted b_hat_svd = A x_svd:")
print(b_hat_svd)

print("\nResidual r_svd = b - A x_svd:")
print(residual_svd)
print(f"\n2-norm of residual ||r_svd||_2 = {res_norm_svd:.6e}")

# Compare with previous solutions from (b) and (c)
diff_svd_ls = np.linalg.norm(x_svd - x_ls, ord=2)
diff_svd_normal = np.linalg.norm(x_svd - x_normal, ord=2)

print("\n=== Comparison with previous solutions ===")
print(f"||x_svd - x_ls||_2      = {diff_svd_ls:.6e}")
print(f"||x_svd - x_normal||_2  = {diff_svd_normal:.6e}")
print("\n(SVD-based solution should agree with lstsq and normal equations up to roundoff.)\n")
#%% (e) Least-squares solution using the built-in pseudo-inverse A^+
print("=== (e) Least-squares solution using the built-in pseudo-inverse A^+ ===")

# Compute pseudo-inverse via NumPy's Moore–Penrose implementation
A_pinv = np.linalg.pinv(A)

print("\nPseudo-inverse A^+ (np.linalg.pinv):")
print(A_pinv)

# Compute solution x = A^+ b
x_pinv = A_pinv @ b

print("\nBuilt-in pseudo-inverse solution x_pinv:")
print(x_pinv)

# Compute predicted b_hat and residual
b_hat_pinv = A @ x_pinv
residual_pinv = b - b_hat_pinv
res_norm_pinv = np.linalg.norm(residual_pinv, ord=2)

print("\nPredicted b_hat_pinv = A x_pinv:")
print(b_hat_pinv)

print("\nResidual r_pinv = b - A x_pinv:")
print(residual_pinv)
print(f"\n2-norm of residual ||r_pinv||_2 = {res_norm_pinv:.6e}")

# Compare with previous solutions (lstsq, normal equations, SVD-based)
diff_pinv_ls = np.linalg.norm(x_pinv - x_ls, ord=2)
diff_pinv_normal = np.linalg.norm(x_pinv - x_normal, ord=2)
diff_pinv_svd = np.linalg.norm(x_pinv - x_svd, ord=2)

print("\n=== Comparison with previous solutions ===")
print(f"||x_pinv - x_ls||_2      = {diff_pinv_ls:.6e}")
print(f"||x_pinv - x_normal||_2  = {diff_pinv_normal:.6e}")
print(f"||x_pinv - x_svd||_2     = {diff_pinv_svd:.6e}")
print("\n(All four solutions should agree up to numerical roundoff.)\n")
#%% (f) Least-squares solution using SVD in the form x = V Σ^+ U^T b
print("=== (f) Least-squares solution using SVD: x = V Σ^+ U^T b ===")

# Compute reduced SVD: A = U Σ V^T
U, s, Vt = np.linalg.svd(A, full_matrices=False)

print("\nSingular values s:")
print(s)

# Step 1: y = U^T b
y = U.T @ b
print("\nVector y = U^T b:")
print(y)

# Step 2: z_i = y_i / s_i  (handle tiny singular values with a tolerance)
tol = 1e-14
z = np.zeros_like(y)
for i in range(len(s)):
    if s[i] > tol:
        z[i] = y[i] / s[i]
    else:
        z[i] = 0.0

print("\nVector z = Σ^+ y (component-wise division y_i / s_i):")
print(z)

# Step 3: x_svd_direct = V z
V = Vt.T
x_svd_direct = V @ z

print("\nSVD-based least-squares solution x_svd_direct:")
print(x_svd_direct)

# Predicted b and residual
b_hat_svd_direct = A @ x_svd_direct
residual_svd_direct = b - b_hat_svd_direct
res_norm_svd_direct = np.linalg.norm(residual_svd_direct, ord=2)

print("\nPredicted b_hat_svd_direct = A x_svd_direct:")
print(b_hat_svd_direct)

print("\nResidual r_svd_direct = b - A x_svd_direct:")
print(residual_svd_direct)
print(f"\n2-norm of residual ||r_svd_direct||_2 = {res_norm_svd_direct:.6e}")

# Compare with previous solutions (lstsq, normal, SVD A^+, pinv)
diff_svd_direct_ls      = np.linalg.norm(x_svd_direct - x_ls, ord=2)
diff_svd_direct_normal  = np.linalg.norm(x_svd_direct - x_normal, ord=2)
diff_svd_direct_svdpinv = np.linalg.norm(x_svd_direct - x_svd, ord=2)
diff_svd_direct_pinv    = np.linalg.norm(x_svd_direct - x_pinv, ord=2)

print("\n=== Comparison with previous solutions ===")
print(f"||x_svd_direct - x_ls||_2      = {diff_svd_direct_ls:.6e}")
print(f"||x_svd_direct - x_normal||_2  = {diff_svd_direct_normal:.6e}")
print(f"||x_svd_direct - x_svd||_2     = {diff_svd_direct_svdpinv:.6e}")
print(f"||x_svd_direct - x_pinv||_2    = {diff_svd_direct_pinv:.6e}")
print("\n(All five solutions should agree up to numerical roundoff.)\n")

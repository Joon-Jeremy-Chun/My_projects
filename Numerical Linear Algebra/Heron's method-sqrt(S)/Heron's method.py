#%% Purpose
# This script implements Heron's method (Babylonian method) for approximating sqrt(S).
# It computes iterates, prints numerical results, and saves plots in the "figures/" directory.

#%% 0. Setup: import libraries and define helper function
import numpy as np
import matplotlib.pyplot as plt
import os

# Create folder for saving figures
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

def heron_sqrt(S, x0, max_iter):
    """
    Perform Heron's method iterations to approximate sqrt(S).

    Parameters
    ----------
    S : float
        Positive number inside the square root.
    x0 : float
        Initial guess.
    max_iter : int
        Number of iterations.

    Returns
    -------
    xs : numpy.ndarray
        Array containing all iterates x_n.
    """
    xs = np.zeros(max_iter + 1)
    xs[0] = x0
    for n in range(max_iter):
        xs[n+1] = 0.5 * (xs[n] + S / xs[n])
    return xs


#%% 1. (a) Compute iterates and print results
S = 2.0
x0 = 1.0
max_iter = 10

xs = heron_sqrt(S, x0, max_iter)
true_sqrt = np.sqrt(S)

print(f"Heron's method approximation for sqrt({S})")
print("-" * 50)
print(f"{'n':>3s} | {'x_n (approx)':>15s} | {'error':>20s}")
print("-" * 50)

for n, x_n in enumerate(xs):
    err = abs(x_n - true_sqrt)
    print(f"{n:3d} | {x_n:15.10f} | {err:20.10e}")

print("-" * 50)
print(f"True sqrt({S}) = {true_sqrt:.10f}")


#%% 2. (b) Error plots: regular plot and semilogy plot
# Compute error values
errors = np.abs(xs - true_sqrt)
n_vals = np.arange(len(xs))

# Regular plot of the error
plt.figure(figsize=(6, 4))
plt.plot(n_vals, errors, marker='o')
plt.xlabel("Iteration n")
plt.ylabel("Error |x_n - sqrt(S)|")
plt.title("Error Plot (Heron's Method)")
plt.grid(True)
plt.savefig(os.path.join(FIG_DIR, "error_plot_regular.png"), dpi=300)
plt.show()

# Semilogy plot of the error
plt.figure(figsize=(6, 4))
plt.semilogy(n_vals, errors, marker='o')
plt.xlabel("Iteration n")
plt.ylabel("Error |x_n - sqrt(S)| (log scale)")
plt.title("Semilogy Error Plot (Heron's Method)")
plt.grid(True, which="both")
plt.savefig(os.path.join(FIG_DIR, "error_plot_semilogy.png"), dpi=300)
plt.show()


#%% 3. (c) Log-log plot of the error
plt.figure(figsize=(6, 4))
# Avoid plotting n=0 if error_0 is too large or zero; here we include all for completeness
plt.loglog(n_vals[1:], errors[1:], marker='o')
plt.xlabel("Iteration n (log scale)")
plt.ylabel("Error |x_n - sqrt(S)| (log scale)")
plt.title("Log-Log Error Plot (Heron's Method)")
plt.grid(True, which="both")
plt.savefig(os.path.join(FIG_DIR, "error_plot_loglog.png"), dpi=300)
plt.show()


#%% 4. (d) Relative error: compute and plot
# Relative error: |x_n - sqrt(S)| / |sqrt(S)|
rel_errors = errors / abs(true_sqrt)

print("Relative errors:")
print("-" * 40)
print(f"{'n':>3s} | {'relative error':>20s}")
print("-" * 40)
for n, re in enumerate(rel_errors):
    print(f"{n:3d} | {re:20.10e}")
print("-" * 40)

# Plot relative error (semilogy)
plt.figure(figsize=(6, 4))
plt.semilogy(n_vals, rel_errors, marker='o')
plt.xlabel("Iteration n")
plt.ylabel("Relative error")
plt.title("Semilogy Plot of Relative Error")
plt.grid(True, which="both")
plt.savefig(os.path.join(FIG_DIR, "relative_error_semilogy.png"), dpi=300)
plt.show()


#%% 5. (f) First iteration where error is below machine precision
eps = np.finfo(float).eps
indices = np.where(errors <= eps)[0]

print(f"Machine epsilon (float) = {eps:.3e}")
if len(indices) > 0:
    first_idx = int(indices[0])
    print(f"First n such that error < eps: n = {first_idx}")
    print(f"Error at n = {first_idx}: {errors[first_idx]:.3e}")
else:
    print("No iteration satisfies error < machine epsilon for the chosen max_iter.")


#%% 6. (extra) Comparison for different initial guesses
initial_guesses = [0.5, 1.0, 3.0, 10.0]
tol = 1e-12

print("\nComparison of iterations needed for different initial guesses:")
print("-" * 60)
print(f"{'x0':>8s} | {'iterations to reach tol':>26s}")
print("-" * 60)

plt.figure(figsize=(6, 4))

for x0_val in initial_guesses:
    xs_k = heron_sqrt(S, x0_val, max_iter)
    errors_k = np.abs(xs_k - true_sqrt)

    # Find the first iteration where error < tol
    idxs = np.where(errors_k < tol)[0]
    if len(idxs) > 0:
        k = int(idxs[0])
    else:
        k = None

    if k is not None:
        print(f"{x0_val:8.3f} | {k:26d}")
    else:
        print(f"{x0_val:8.3f} | {'did not reach tol':>26s}")

    # Plot error curves for each initial guess
    plt.semilogy(np.arange(len(errors_k)), errors_k, marker='o', label=f"x0 = {x0_val}")

print("-" * 60)

plt.xlabel("Iteration n")
plt.ylabel("Error |x_n - sqrt(S)| (log scale)")
plt.title("Semilogy Error Plot for Different Initial Guesses")
plt.grid(True, which="both")
plt.legend()
plt.savefig(os.path.join(FIG_DIR, "error_semilogy_initial_guesses.png"), dpi=300)
plt.show()

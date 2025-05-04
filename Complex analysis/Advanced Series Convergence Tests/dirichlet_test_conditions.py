import os
import numpy as np
import matplotlib.pyplot as plt

# Create output directory
output_dir = os.path.join("Figures", "dirichlet_test_conditions")
os.makedirs(output_dir, exist_ok=True)

# Parameters
N = 50
n = np.arange(1, N+1)

# Sequence a_n for Dirichlet test: alternating +/-1
a_n = (-1)**(n+1)
# Partial sums A_n = sum_{k=1}^n a_k
A_n = np.cumsum(a_n)

# Sequence b_n: monotonic decreasing to zero
b_n = 1 / n

# Plot bounded partial sums
plt.figure(figsize=(6,4))
plt.step(n, A_n, where='post', label=r"$A_n = \sum_{k=1}^n a_k,\ a_k = (-1)^{k+1}$")
plt.hlines([0,1], xmin=1, xmax=N, colors='gray', linestyles=':', 
           label="Bounds of $A_n$")
plt.title("Bounded Partial Sums for Dirichlet Test Condition")
plt.xlabel("n")
plt.ylabel(r"$A_n$")
plt.xlim(1, N)
plt.ylim(-0.5, 1.5)
plt.legend()
plt.grid(True)
partial_path = os.path.join(output_dir, "bounded_partial_sums.png")
plt.savefig(partial_path, dpi=300)

# Plot monotonic decreasing b_n
plt.figure(figsize=(6,4))
plt.plot(n, b_n, 'o-', label=r"$b_n = 1/n$")
plt.title("Monotonic Decreasing Sequence for Dirichlet Test Condition")
plt.xlabel("n")
plt.ylabel(r"$b_n$")
plt.xscale('log')
plt.yscale('log')
plt.xlim(1, N)
plt.ylim(1/N, 1.1)
plt.legend()
plt.grid(True, which='both', ls=':')
bn_path = os.path.join(output_dir, "monotone_decreasing_bn.png")
plt.savefig(bn_path, dpi=300)

# Display figures
plt.show()

# Print saved paths
print("Saved bounded partial sums to:", os.path.abspath(partial_path))
print("Saved monotonic decreasing b_n to:", os.path.abspath(bn_path))

import os
import numpy as np
import matplotlib.pyplot as plt

def a(n):
    """
    Sequence definition:
      if n is even: a_n = 1 / 2^n
      if n is odd : a_n = 1 / 2^(n+1)
    """
    return 1 / (2**n) if n % 2 == 0 else 1 / (2**(n+1))

# Create output directory
output_dir = os.path.join("Figures", "ratio_test_vs_root_test")
os.makedirs(output_dir, exist_ok=True)

# Parameters
N = 50

# Compute Ratio test, Root test, and partial sums
ratio_values   = [abs(a(n+1) / a(n))     for n in range(1, N)]
root_values    = [a(n)**(1/n)            for n in range(1, N)]
partial_sums   = []
cum_sum = 0.0
for n in range(1, N):
    cum_sum += a(n)
    partial_sums.append(cum_sum)

# 1) Ratio Test Figure
plt.figure(figsize=(6,4))
plt.plot(range(1, N), ratio_values, 'o--', label=r"$\left|\frac{a_{n+1}}{a_n}\right|$")
plt.axhline(1.0, color='gray', linestyle=':', label="Limit = 1")
plt.title("Ratio Test Values")
plt.xlabel("n")
plt.ylabel("Ratio")
plt.legend()
plt.grid(True)
ratio_path = os.path.join(output_dir, "ratio_test.png")
plt.savefig(ratio_path, dpi=300)

# 2) Root Test Figure
plt.figure(figsize=(6,4))
plt.plot(range(1, N), root_values, 's--', label=r"$\sqrt[n]{a_n}$")
plt.axhline(0.5, color='gray', linestyle=':', label="Limit = 0.5")
plt.title("Root Test Values")
plt.xlabel("n")
plt.ylabel("Root")
plt.legend()
plt.grid(True)
root_path = os.path.join(output_dir, "root_test.png")
plt.savefig(root_path, dpi=300)

# 3) Actual Partial Sums Figure
plt.figure(figsize=(6,4))
plt.plot(range(1, N), partial_sums, 'o-', color='tab:green', label=r"$S_n = \sum_{k=1}^n a_k$")
plt.title("Actual Partial Sums of $a_n$")
plt.xlabel("n")
plt.ylabel("S_n")
plt.legend()
plt.grid(True)
sums_path = os.path.join(output_dir, "actual_sums.png")
plt.savefig(sums_path, dpi=300)

# Optionally display all figures
plt.show()

# Print saved file paths
print("Saved ratio test figure to:", os.path.abspath(ratio_path))
print("Saved root test figure to: ", os.path.abspath(root_path))
print("Saved partial sums figure to:", os.path.abspath(sums_path))

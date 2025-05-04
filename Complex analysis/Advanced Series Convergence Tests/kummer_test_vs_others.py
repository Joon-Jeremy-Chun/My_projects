import os
import math
import matplotlib.pyplot as plt

def double_factorial(n):
    """Compute the double factorial n!!."""
    if n <= 0:
        return 1
    result = 1
    while n > 0:
        result *= n
        n -= 2
    return result

def a(n, p):
    """
    nth term of the series for given p:
      a_n = [ (2n-1)!! / (2n)!! ]^p
    """
    base = double_factorial(2*n - 1) / double_factorial(2*n)
    return base ** p

# Create output directory
output_dir = os.path.join("Figures", "kummer_test_vs_others")
os.makedirs(output_dir, exist_ok=True)

# Parameters
p = 2
N = 50

# Compute test values
ratio_values = [a(n+1, p) / a(n, p) for n in range(1, N)]
raabe_values = [n * (a(n, p) / a(n+1, p) - 1) for n in range(1, N)]
kummer_values = []
for n in range(1, N):
    Bn = n * math.log(n)
    Bn1 = (n + 1) * math.log(n + 1)
    kummer_values.append(Bn * (a(n, p) / a(n+1, p)) - Bn1)

partial_sums = []
cum = 0.0
for n in range(1, N):
    cum += a(n, p)
    partial_sums.append(cum)

# 1) Ratio Test Plot
plt.figure(figsize=(6,4))
plt.plot(range(1, N), ratio_values, marker='o', linestyle='--', label=r"$\frac{a_{n+1}}{a_n}$")
plt.axhline(1, linestyle=':', label="Limit = 1")
plt.title(f"Ratio Test (p={p})")
plt.xlabel("n")
plt.ylabel("Ratio")
plt.grid(True)
plt.legend()
ratio_path = os.path.join(output_dir, "ratio_test.png")
plt.savefig(ratio_path, dpi=300)

# 2) Raabe's Test Plot
plt.figure(figsize=(6,4))
plt.plot(range(1, N), raabe_values, marker='s', linestyle='--', label=r"$n\left(\frac{a_n}{a_{n+1}}-1\right)$")
plt.axhline(p/2, linestyle=':', label=f"Limit = {p/2}")
plt.title(f"Raabe's Test (p={p})")
plt.xlabel("n")
plt.ylabel("Raabe's value")
plt.grid(True)
plt.legend()
raabe_path = os.path.join(output_dir, "raabe_test.png")
plt.savefig(raabe_path, dpi=300)

# 3) Kummer's Test Plot
plt.figure(figsize=(6,4))
plt.plot(range(1, N), kummer_values, marker='^', linestyle='--',
         label=r"$B_n\frac{a_n}{a_{n+1}} - B_{n+1}$")
plt.axhline(0, linestyle=':', label="Limit = 0")
plt.title(f"Kummer's Test (p={p})")
plt.xlabel("n")
plt.ylabel("Kummer's value")
plt.grid(True)
plt.legend()
kummer_path = os.path.join(output_dir, "kummer_test.png")
plt.savefig(kummer_path, dpi=300)

# 4) Partial Sums Plot
plt.figure(figsize=(6,4))
plt.plot(range(1, N), partial_sums, marker='o', linestyle='-',
         label=r"$S_n = \sum_{k=1}^n a_k$")
plt.title(f"Partial Sums (p={p})")
plt.xlabel("n")
plt.ylabel("S_n")
plt.grid(True)
plt.legend()
sums_path = os.path.join(output_dir, "partial_sums.png")
plt.savefig(sums_path, dpi=300)

# Display all figures
plt.show()

# Print saved file paths
print("Saved:", os.path.abspath(ratio_path))
print("Saved:", os.path.abspath(raabe_path))
print("Saved:", os.path.abspath(kummer_path))
print("Saved:", os.path.abspath(sums_path))

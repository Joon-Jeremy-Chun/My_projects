import os
import matplotlib.pyplot as plt

def double_factorial(n):
    if n <= 0:
        return 1
    result = 1
    while n > 0:
        result *= n
        n -= 2
    return result

def a(n):
    num = double_factorial(2*n - 1)
    den = double_factorial(2*n)
    return (num / den) * (1 / (2*n + 1))

# Ensure output directory exists
output_dir = os.path.join("Figures", "Raabe's Test vs other's")
os.makedirs(output_dir, exist_ok=True)

# Parameters
N = 50

# Compute test values
ratio_values = [a(n+1) / a(n) for n in range(1, N)]
root_values = [a(n) ** (1 / n) for n in range(1, N)]
raabe_values = [n * (a(n) / a(n+1) - 1) for n in range(1, N)]

# Compute partial sums (series)
partial_sums = []
cum = 0
for n in range(1, N):
    cum += a(n)
    partial_sums.append(cum)

# 1) Ratio Test
plt.figure()
plt.plot(range(1, N), ratio_values, 'o--', label=r"$\left|\frac{a_{n+1}}{a_n}\right|$")
plt.axhline(1.0, linestyle=':', label="Limit = 1")
plt.title(r"Ratio Test: $a_n = \frac{(2n-1)!!}{(2n)!!}\cdot\frac{1}{2n+1}$")
plt.xlabel("n")
plt.ylabel(r"Ratio $\left(\frac{a_{n+1}}{a_n}\right)$")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "ratio_test.png"), dpi=300)

# 2) Root Test
plt.figure()
plt.plot(range(1, N), root_values, 'x--', label=r"$\sqrt[n]{a_n}$")
plt.axhline(1.0, linestyle=':', label="Limit = 1")
plt.title(r"Root Test: $a_n = \frac{(2n-1)!!}{(2n)!!}\cdot\frac{1}{2n+1}$")
plt.xlabel("n")
plt.ylabel(r"Root value $\sqrt[n]{a_n}$")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "root_test.png"), dpi=300)

# 3) Raabe's Test
plt.figure()
plt.plot(range(1, N), raabe_values, 's--', label=r"$n\left(\frac{a_n}{a_{n+1}} - 1\right)$")
plt.axhline(1.5, linestyle=':', label="Limit = 1.5")
plt.title(r"Raabe's Test: $a_n = \frac{(2n-1)!!}{(2n)!!}\cdot\frac{1}{2n+1}$")
plt.xlabel("n")
plt.ylabel("Raabe's value")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "raabe_test.png"), dpi=300)

# 4) Partial Sums (Series)
plt.figure()
plt.plot(range(1, N), partial_sums, 'o-', label=r"$S_n = \sum_{k=1}^n a_k$")
plt.title(r"Partial Sums: $S_n = \sum_{k=1}^n a_k$")
plt.xlabel("n")
plt.ylabel(r"$S_n$")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "partial_sums.png"), dpi=300)

plt.show()

# Print saved file paths
print("Saved files:")
print(os.path.abspath(os.path.join(output_dir, "ratio_test.png")))
print(os.path.abspath(os.path.join(output_dir, "root_test.png")))
print(os.path.abspath(os.path.join(output_dir, "raabe_test.png")))
print(os.path.abspath(os.path.join(output_dir, "partial_sums.png")))

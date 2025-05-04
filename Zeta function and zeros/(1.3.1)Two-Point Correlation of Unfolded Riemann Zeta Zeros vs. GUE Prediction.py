import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------------------------------------------------------
# 2-Point Correlation of Unfolded Riemann Zeta Zeros vs. GUE Prediction
# -----------------------------------------------------------------------------

# 1. Setup
mp.mp.dps = 50                             # precision for zetazero
fig_dir = 'Figures'
os.makedirs(fig_dir, exist_ok=True)

# 2. Compute the first 100 nontrivial zeros γₙ on the critical line
zeros = np.array([mp.zetazero(n).imag for n in range(1, 101)], dtype=float)

# 3. Unfolding: rescale spacings so that mean spacing = 1
raw_spacings = np.diff(zeros)             # γₙ₊₁ − γₙ
mean_spacing = np.mean(raw_spacings)
normalized_spacings = raw_spacings / mean_spacing
# build unfolded positions \tilde{γₙ}
unfolded_positions = np.concatenate([[0], np.cumsum(normalized_spacings)])

# 4. Compute all positive pairwise differences Δ = \tilde{γ_j} - \tilde{γ_i}, j>i
N = len(unfolded_positions)
pair_diffs = []
for i in range(N):
    for j in range(i+1, N):
        pair_diffs.append(unfolded_positions[j] - unfolded_positions[i])
pair_diffs = np.array(pair_diffs)

# 5. Empirical 2-point correlation: histogram of pairwise differences
bins = np.linspace(0, 10, 200)            # up to ξ=10
hist_vals, bin_edges = np.histogram(pair_diffs, bins=bins, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 6. GUE prediction: R2_GUE(ξ) = 1 – (sin(πξ)/(πξ))^2
def R2_gue(x):
    return 1.0 - (np.sin(np.pi*x) / (np.pi*x))**2

xs = np.linspace(0.01, 10, 500)           # avoid ξ=0 for division

# 7. Plot comparison
plt.figure(figsize=(8, 6))
plt.bar(bin_centers, hist_vals, width=bin_edges[1]-bin_edges[0],
        alpha=0.6, label='Empirical $R_2(\\xi)$')
plt.plot(xs, R2_gue(xs), 'r--', lw=2, label='GUE $R_2(\\xi)$')
plt.xlabel('Distance $\\xi$')
plt.ylabel('Correlation $R_2(\\xi)$')
plt.title('Two-Point Correlation of Unfolded Zeta Zeros vs. GUE')
plt.legend()
plt.grid(True)
plt.tight_layout()

# 8. Save figure
output_path = os.path.join(fig_dir, 'r2_correlation_comparison.png')
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Saved two-point correlation plot to: {output_path}")

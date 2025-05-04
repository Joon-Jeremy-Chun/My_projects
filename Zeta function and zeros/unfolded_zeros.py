import os
import math
import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt

# Ensure the Figures directory exists
figures_dir = "Figures"
os.makedirs(figures_dir, exist_ok=True)

# Step 1: Compute nontrivial zeros γ_n of ζ(½ + iγ)
mp.mp.dps = 80  # precision

def zeta_half_it(t):
    return mp.zeta(0.5 + 1j*t)

def find_zero_near(t0, delta=0.5):
    root = mp.findroot(zeta_half_it, [t0 - delta, t0 + delta])
    return float(mp.re(root))

initial_guesses = [14, 21, 25, 30, 32, 37, 40, 43, 48, 49]
zeros = [find_zero_near(g) for g in initial_guesses]

# Step 2: Unfold zeros
def approximate_zero_count(t):
    if t <= 2 * math.pi:
        return 0.0
    return (t/(2*math.pi))*math.log(t/(2*math.pi)) - (t/(2*math.pi))

unfolded = np.array([approximate_zero_count(t) for t in zeros])

# Step 3: Compute spacings
spacings = np.diff(unfolded)

# Step 4: Plot and save histogram vs. GUE distribution
plt.figure(figsize=(8,5))
n, bins, patches = plt.hist(spacings, bins=15, density=True, alpha=0.6, label="Empirical spacings")

x = np.linspace(0, 4, 200)
gue_pdf = (32/np.pi**2) * x**2 * np.exp(-4*x**2/np.pi)
plt.plot(x, gue_pdf, 'r-', lw=2, label="GUE Wigner–Dyson PDF")

plt.xlabel("Spacing (unfolded)")
plt.ylabel("Probability density")
plt.title("Nearest-Neighbor Spacings of Zeta Zeros vs. GUE")
plt.legend()
plt.grid(True)

# Save the figure
histogram_path = os.path.join(figures_dir, "spacings_vs_GUE.png")
plt.savefig(histogram_path)
plt.close()

# Optional: you can create and save more figures similarly:
# e.g., a plot of unfolded zeros
plt.figure(figsize=(6,4))
plt.plot(unfolded, 'o-')
plt.xlabel("Index n")
plt.ylabel("Unfolded coordinate x_n")
plt.title("Unfolded Zeta Zeros")
unfolded_path = os.path.join(figures_dir, "unfolded_zeros.png")
plt.savefig(unfolded_path)

# Display saved file paths for confirmation
histogram_path, unfolded_path

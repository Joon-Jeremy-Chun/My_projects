import os
import numpy as np
import matplotlib.pyplot as plt

# 1) Setup output directory
output_dir = os.path.join("Figures", "complex_convergence")
os.makedirs(output_dir, exist_ok=True)

# 2.1) Complex plane region: Re(z)>1 absolute convergence region
fig, ax = plt.subplots(figsize=(6,6))
# Shade region Re(z)>1
ax.axvspan(1, 3, color='lightgray', alpha=0.5)
# Plot and annotate points
points = [
    (1.5, -0.5, "z = 3/2 − 1/2 i\n(absolute)"),
    (0.5,  3.0, "z = 1/2 + 3 i\n(conditional)"),
]
for x, y, label in points:
    ax.plot(x, y, 'o', markersize=8, color='tab:blue')
    ax.annotate(label, (x, y), textcoords="offset points", xytext=(5,5))
# Axes settings
ax.set_xlim(-1, 3)
ax.set_ylim(-4, 4)
ax.set_xlabel("Re(z)")
ax.set_ylabel("Im(z)")
ax.set_title("Absolute vs. Conditional Convergence Region")
ax.axvline(1, color='black', linestyle='--', linewidth=1)
ax.text(1, 4.1, "Re(z)=1", ha="center")
ax.grid(True)
region_path = os.path.join(output_dir, "convergence_region.png")
plt.savefig(region_path, dpi=300)

# 2.2) Decay of |1/n^z| and partial sums for z1, z2
N = 100
n = np.arange(1, N)
# Real parts
re1 = 1.5  # absolute convergence
re2 = 0.5  # conditional convergence
v1 = n ** (-re1)
v2 = n ** (-re2)
s1 = np.cumsum(v1)
s2 = np.cumsum(v2)

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(6,8), sharex=True)

# Decay plot
ax1.loglog(n, v1, label=r"$|1/n^z|$ for Re(z)=3/2", marker='o', linestyle='--')
ax1.loglog(n, v2, label=r"$|1/n^z|$ for Re(z)=1/2", marker='s', linestyle='--')
ax1.set_ylabel("Term magnitude")
ax1.set_title("Decay of |1/n^z|")
ax1.legend()
ax1.grid(True, which="both", ls=":")

# Partial sums plot
ax2.plot(n, s1, label=r"$\sum 1/n^{3/2}$ (converges)", marker='o')
ax2.plot(n, s2, label=r"$\sum 1/n^{1/2}$ (diverges)", marker='s')
ax2.set_xlabel("n")
ax2.set_ylabel("Partial sum")
ax2.set_title("Partial Sums of |1/n^z|")
ax2.legend()
ax2.grid(True)

decay_path = os.path.join(output_dir, "decay_and_sums.png")
plt.tight_layout()
plt.savefig(decay_path, dpi=300)

plt.show()

# Print saved paths
print("Saved convergence region to:", os.path.abspath(region_path))
print("Saved decay & sums plot to:", os.path.abspath(decay_path))

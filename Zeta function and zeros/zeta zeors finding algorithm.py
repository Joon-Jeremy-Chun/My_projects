# -*- coding: utf-8 -*-
"""
Robust search for the first 50 Riemann zeta zeros on the critical line,
plus unfolding and plotting both raw vs unfolded.
Saves figure in Figures/raw_vs_unfolded_50_actual.png
"""
import os, math
import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt

# 0) Prepare output directory
figures_dir = "Figures"
os.makedirs(figures_dir, exist_ok=True)

# 1) Setup precision and zeta‐half‐it function
mp.mp.dps = 50
def zeta_half_it(t):
    return mp.zeta(0.5 + 1j*t)

def find_zero_near(guess, deltas=(0.5, 1.0, 2.0, 5.0)):
    """
    Try to find a zero of ζ(½ + i t) near 'guess' using progressively wider brackets.
    Returns a real-valued root if found, else raises.
    """
    for delta in deltas:
        try:
            a, b = guess-delta, guess+delta
            root = mp.findroot(zeta_half_it, [a, b])
            return float(mp.re(root))
        except Exception:
            pass
    raise ValueError(f"Could not bracket a root near t≈{guess}")

# 2) Produce approximate initial guesses via main‐term Riemann–von Mangoldt
def initial_guess(n):
    return (2*math.pi*n) / math.log(2*math.pi*n)

# 3) Find up to 50 zeros
zeros = []
print("Locating zeros n=1..50:")
for n in range(1, 51):
    g = initial_guess(n)
    try:
        gamma_n = find_zero_near(g)
        zeros.append(gamma_n)
        print(f"  n={n:2d}: γ ≈ {gamma_n:.8f}")
    except ValueError as e:
        print(f"  n={n:2d}: FAIL ({e})")
        # we still move on; we'll end up with fewer than 50 if some fail

zeros = np.array(zeros)
if len(zeros) < 2:
    raise RuntimeError("Not enough zeros found to proceed.")

# 4) Unfold via main‐term von Mangoldt
def approx_count(t):
    if t <= 2*math.pi:
        return 0.0
    return (t/(2*math.pi))*math.log(t/(2*math.pi)) - (t/(2*math.pi))

unfolded = np.array([approx_count(t) for t in zeros])

# 5) Plot raw vs unfolded
n = np.arange(1, len(zeros)+1)
plt.figure(figsize=(10,5))
plt.plot(n, zeros,   'o-', label=r'Original $\gamma_n$', markersize=4)
plt.plot(n, unfolded,'s--', label=r'Unfolded $x_n$',   markersize=4)
plt.xlabel('Index $n$')
plt.ylabel('Value')
plt.title(f'Original vs Unfolded Zeta Zeros (found {len(zeros)} of 50)')
plt.legend()
plt.grid(True)
plt.tight_layout()

out_path = os.path.join(figures_dir, "raw_vs_unfolded_50_actual.png")
plt.savefig(out_path)
plt.show()

print(f"\nSaved comparison plot to {out_path}")

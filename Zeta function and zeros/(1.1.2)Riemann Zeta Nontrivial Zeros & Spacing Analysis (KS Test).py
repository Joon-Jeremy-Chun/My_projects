import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import kstest

# Ensure Figures directory exists
fig_dir = 'Figures'
os.makedirs(fig_dir, exist_ok=True)

# ------------------------------
# Set precision and define input zeros
# ------------------------------
mp.mp.dps = 80  # high precision

# List of first 100 imaginary parts to locate
initial_guesses = [
    14.1347251417347, 21.0220396387716, 25.0108575801457,
    30.4248761258595, 32.9350615877392, 37.5861781588257,
    40.9187190121475, 43.3270732809149, 48.0051508811672,
    49.7738324776723, 52.9703214777144, 56.4462476970634,
    59.3470440026026, 60.8317785246098, 65.1125440480819,
    67.0798105294941, 69.5464017111739, 72.0671576744819,
    75.7046906990839, 77.1448400688748, 79.3373750202494,
    82.9103808540860, 84.7354929805171, 87.4252746131252,
    88.8091112076345, 92.4918992705585, 94.6513440405199,
    95.8706342282453, 98.8311942181937, 101.3178510057310,
    103.7255380404780, 105.4466230523260, 107.1686111842760,
    111.0295355431700, 111.8746591769930, 114.3202209154520,
    116.2266803208570, 118.7907828659760, 121.3701250021460,
    122.9468292931260, 124.2568185543450, 127.5166838790360,
    129.5787042007380, 131.0876885309320, 133.4977372020070,
    134.7565097533730, 138.1160420545330, 139.7362089521210,
    141.1237074040220, 143.1118458079810, 146.0009824873420,
    147.4227653436520, 150.0535204212080, 150.9252576120680,
    153.0246938110140, 156.1129092940780, 157.5975918176410,
    158.8499881710660, 161.1889641375960, 163.0307096871810,
    165.5370698495930, 167.1844399785810, 169.0945154154490,
    169.9119764797110, 173.4115365195220, 175.2756873354450,
    176.4414342977100, 178.3774077769950, 179.9164840202210,
    182.2070784845470, 184.8744678481030, 185.5987836777080,
    187.2289225847540, 189.4161586566790, 192.0266563612960,
    193.0797266038310, 195.2653966795860, 196.8764818413990,
    198.0153096768130, 201.2647519441090, 202.4935945141760,
    204.1896718036820, 205.3946972021320, 207.9062588874820,
    209.5765097166670, 211.6908625950190, 213.3479191357050,
    214.5470447832990, 216.1695385085580, 219.0675963490220,
    220.7149188690170, 221.4307055544710, 224.0070006665970,
    224.9833241510440, 227.4214442792610, 229.3374133060490,
    231.2501887004420, 232.4977532058540, 234.3202096637240,
    236.5242296662600
]

def find_zero_near(guess):
    s0 = mp.mpc(0.5, guess)
    try:
        # Newton’s method, catch ValueError on failure
        return mp.findroot(lambda s: mp.zeta(s), s0, tol=1e-12, maxsteps=100)
    except ValueError:
        # Fallback: secant with two starting points
        s1 = s0 + mp.mpc(0, 0.1)
        try:
            return mp.findroot(lambda s: mp.zeta(s), [s0, s1], tol=1e-12, maxsteps=100)
        except Exception as e:
            raise RuntimeError(f"Root finding failed at γ≈{guess}: {e}")

# ------------------------------
# Compute zeros robustly
# ------------------------------
zeros = []
for g in initial_guesses:
    try:
        root = find_zero_near(g)
        zeros.append(float(root.imag))
    except RuntimeError as err:
        print(err)
        # skip this guess

# ------------------------------
# Analyze zero spacings with KS test
# ------------------------------
spacings = np.diff(zeros)
D_stat, p_value = kstest(spacings, 'expon', args=(0, np.mean(spacings)))

# ------------------------------
# Plot results
# ------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(range(1, len(zeros)+1), zeros, 'o-')
ax1.set_title("Imaginary Parts of Zeta Zeros (γₙ)")
ax1.set_xlabel("Index n")
ax1.set_ylabel("γₙ")
ax1.grid(True)

ax2.hist(spacings, bins=10, density=True, alpha=0.7, edgecolor='black')
ax2.set_title("Histogram of Zero Spacings Δγₙ")
ax2.set_xlabel("Spacing Δγ")
ax2.set_ylabel("Density")
ax2.grid(True)

plt.tight_layout()

# Save and show
output_path = os.path.join(fig_dir, 'zeta_zeros_and_spacings.png')
fig.savefig(output_path, dpi=300)
plt.show()

# Summary
print(f"KS test D statistic = {D_stat:.4f}, p-value = {p_value:.4f}")
print(f"Saved figure to: {output_path}")

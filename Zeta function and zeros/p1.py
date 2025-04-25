import math
import numpy as np
import matplotlib.pyplot as plt

def approximate_zero_count(t):
    """
    Approximate how many nontrivial Riemann zeros lie below imaginary part t
    using the main term of the Riemann–von Mangoldt formula:
      N(t) ~ (t / (2 pi)) * log(t / (2 pi)) - (t / (2 pi)).
    """
    if t <= 2 * math.pi:
        return 0.0
    return (t / (2.0 * math.pi)) * math.log(t / (2.0 * math.pi)) - (t / (2.0 * math.pi))

def unfold_zeros(gammas):
    """
    Given a sorted list 'gammas' of imaginary parts of zeta zeros,
    return a list of unfolded coordinates based on approximate_zero_count(g).
    """
    unfolded = []
    for g in gammas:
        unfolded.append( approximate_zero_count(g) )
    return unfolded

# 1) Provide some dummy or real zeta zeros (sorted) for testing:
zeta_zeros = [
    10001.45, 10002.98, 10005.23, 10007.10, 10009.77
]
# ... In practice, you would replace these with your actual, possibly longer list 
# of computed zeros.

# 2) Unfold the zeros
unfolded_coords = unfold_zeros(zeta_zeros)

# 3) Compute local spacings in the unfolded scale
unfolded_spacings = []
for i in range(len(unfolded_coords) - 1):
    spacing = unfolded_coords[i+1] - unfolded_coords[i]
    unfolded_spacings.append(spacing)

# 4) Convert to NumPy array for convenience
spacings_array = np.array(unfolded_spacings)

# 5) Plot a histogram of the unfolded spacings
plt.hist(spacings_array, bins=20, density=True, alpha=0.6, label="Data")

# 6) Plot the theoretical GUE (Wigner-Dyson) curve for comparison
x_vals = np.linspace(0, 4, 200)
gue_pdf = (32.0 / (np.pi**2)) * (x_vals**2) * np.exp(-4 * x_vals**2 / np.pi)
plt.plot(x_vals, gue_pdf, 'r-', lw=2, label="GUE (Wigner-Dyson)")

plt.xlabel("Spacing (Unfolded)")
plt.ylabel("Density")
plt.title("Unfolded Zero Spacings vs. GUE Distribution")
plt.legend()
plt.show()

def compute_unfolded_spacings(unfolded_coords):
    """
    Given a sorted list of unfolded coordinates, compute the local spacings
    as consecutive differences. Returns a new list of spacings.
    """
    spacings = []
    for i in range(len(unfolded_coords) - 1):
        # spacing = x_{i+1} - x_i
        spacing = unfolded_coords[i+1] - unfolded_coords[i]
        spacings.append(spacing)
    return spacings

# Example usage:
if __name__ == "__main__":
    # Suppose we already have unfolded_coords from step 2.
    unfolded_coords = [0.0, 1.23, 2.47, 3.65, 5.12]  # Dummy example
    
    # Now we compute the local spacings:
    unfolded_spacings = compute_unfolded_spacings(unfolded_coords)
    
    print("Unfolded coordinates:", unfolded_coords)
    print("Unfolded spacings:   ", unfolded_spacings)


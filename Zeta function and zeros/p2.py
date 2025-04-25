import math
import numpy as np
import matplotlib.pyplot as plt

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

# --------------------------------------------------------------------
# Example Usage
# --------------------------------------------------------------------

if __name__ == "__main__":

    # 1) Suppose you have a list of 'unfolded_coords' (the result of applying
    #    some unfolding to your actual zeta zeros).
    #    Here we use dummy data for demonstration:
    unfolded_coords = [0.0, 1.1, 2.4, 3.5, 4.7, 6.1, 7.4, 8.6]

    # 2) Compute local spacings in the unfolded scale
    unfolded_spacings = compute_unfolded_spacings(unfolded_coords)

    # 3) Convert to NumPy array (optional, but handy for plotting)
    spacings_array = np.array(unfolded_spacings)

    # 4) Plot a histogram of the unfolded spacings
    plt.hist(spacings_array, bins=20, density=True, alpha=0.6, label="Unfolded spacings")

    # 5) Compare with the theoretical GUE (Wigner–Dyson) nearest-neighbor distribution:
    x_vals = np.linspace(0, 5, 200)  # Adjust max range as needed
    # GUE Wigner-Dyson PDF for local spacing:
    #   P(s) = (32 / pi^2) * s^2 * exp(-4 s^2 / pi),  s >= 0
    gue_pdf = (32.0 / (np.pi**2)) * (x_vals**2) * np.exp(-4.0 * x_vals**2 / np.pi)
    plt.plot(x_vals, gue_pdf, 'r-', lw=2, label="GUE (Wigner-Dyson)")

    # 6) Finish the plot
    plt.xlabel("Spacing (Unfolded)")
    plt.ylabel("Density")
    plt.title("Unfolded Zero Spacings vs. GUE Distribution")
    plt.legend()
    plt.show()

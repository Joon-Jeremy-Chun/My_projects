# %% Phase Field Plot for Nonlinear Second-Order ODE
# x'' + b x' - k x + x^3 = 0
# System:
#   x' = y
#   y' = k*x - x**3 - b*y

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_phase_field(x_range, y_range, density, b_param, k_param,
                     save_dir="Figures", save_name=None):
    """
    Plot the vector field (phase portrait) for the nonlinear ODE system:
        x' = y
        y' = k*x - x**3 - b*y
    
    Parameters
    ----------
    x_range : tuple (xmin, xmax)
    y_range : tuple (ymin, ymax)
    density : int, number of grid points per axis
    b_param : float, parameter b
    k_param : float, parameter k
    save_dir : folder where the figure will be saved
    save_name : optional custom filename
    """

    # Create directory if missing
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Grid for vector field
    x = np.linspace(x_range[0], x_range[1], density)
    y = np.linspace(y_range[0], y_range[1], density)
    X, Y = np.meshgrid(x, y)

    # Vector field components
    U = Y
    V = k_param*X - X**3 - b_param*Y

    # Normalize for better visualization
    M = np.hypot(U, V)
    U2 = U / (M + 1e-8)
    V2 = V / (M + 1e-8)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.quiver(X, Y, U2, V2, M, cmap='viridis')
    plt.colorbar(label="Vector magnitude")
    plt.xlabel("x")
    plt.ylabel("y")

    title = f"Phase Field: b={b_param}, k={k_param}"
    plt.title(title)

    # Save file name
    if save_name is None:
        save_name = f"phase_field_b={b_param}_k={k_param}.png"

    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Figure saved at: {save_path}")

#%%
plot_phase_field(
    x_range=(-3, 3),
    y_range=(-3, 3),
    density=25,
    b_param=1.0,
    k_param=4.0,
    save_dir="Figures"
)

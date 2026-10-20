#%% 0. Setup: import libraries and define vectors
import numpy as np
import matplotlib.pyplot as plt
import os

# Create folder for saving figures (shared with other problems)
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Given vectors v1, v2, v3
v1 = np.array([-2.0, -1.0, 0.0])
v2 = np.array([-1.0,  2.0, -1.0])
v3 = np.array([ 0.0, -1.0,  2.0])

# #(extra test) vectors v1, v2, v3
# v1 = np.array([1.0, 1.0, 1.0])
# v2 = np.array([1.0 + 1e-17, 1.0, 1.0])   # v1과 거의 평행
# v3 = np.array([0.0, 1.0, -1.0])         # 독립된 벡터
# V = np.column_stack((v1, v2, v3))


# Stack into a matrix V = [v1 v2 v3]
V = np.column_stack((v1, v2, v3))


def classical_gram_schmidt(V):
    """
    Classical Gram-Schmidt (CGS) process.

    Parameters
    ----------
    V : (n, m) array
        Columns are the input vectors.

    Returns
    -------
    Q : (n, m) array
        Orthonormal columns.
    R : (m, m) upper triangular array
        Such that V = Q R (in exact arithmetic).
    """
    n, m = V.shape
    Q = np.zeros((n, m))
    R = np.zeros((m, m))

    for k in range(m):
        v = V[:, k].copy()
        # subtract projections onto previous q_j
        for j in range(k):
            # CGS projects the original column, while MGS updates the working vector.
            R[j, k] = np.dot(Q[:, j], V[:, k])
            v = v - R[j, k] * Q[:, j]
        R[k, k] = np.linalg.norm(v)
        Q[:, k] = v / R[k, k]

    return Q, R


def modified_gram_schmidt(V):
    """
    Modified Gram-Schmidt (MGS) process.

    Parameters
    ----------
    V : (n, m) array

    Returns
    -------
    Q, R : as above.
    """
    n, m = V.shape
    Q = np.zeros((n, m))
    R = np.zeros((m, m))

    # make a working copy of V
    W = V.astype(float).copy()

    for k in range(m):
        # normalize current column
        R[k, k] = np.linalg.norm(W[:, k])
        Q[:, k] = W[:, k] / R[k, k]

        # orthogonalize remaining columns against q_k
        for j in range(k + 1, m):
            R[k, j] = np.dot(Q[:, k], W[:, j])
            W[:, j] = W[:, j] - R[k, j] * Q[:, k]

    return Q, R
#%% 1. (a) Classical Gram-Schmidt
print("=== Classical Gram–Schmidt (CGS) ===")
Q_cgs, R_cgs = classical_gram_schmidt(V)

print("Q_cgs =")
print(Q_cgs)
print("\nR_cgs =")
print(R_cgs)

# Check orthogonality: Q^T Q should be close to I
I_cgs = Q_cgs.T @ Q_cgs
print("\nQ_cgs^T Q_cgs =")
print(I_cgs)
#%% 2. (b) Modified Gram-Schmidt
print("\n=== Modified Gram–Schmidt (MGS) ===")
Q_mgs, R_mgs = modified_gram_schmidt(V)

print("Q_mgs =")
print(Q_mgs)
print("\nR_mgs =")
print(R_mgs)

# Check orthogonality: Q^T Q should be close to I
I_mgs = Q_mgs.T @ Q_mgs
print("\nQ_mgs^T Q_mgs =")
print(I_mgs)

# Quick comparison: difference between Q_cgs and Q_mgs
print("\n||Q_cgs - Q_mgs||_F =", np.linalg.norm(Q_cgs - Q_mgs, ord='fro'))
#%% 3. (c) Plot and save orthogonality error for CGS vs MGS
# Off-diagonal inner products |q_i^T q_j|, i < j
pairs = [(0, 1), (0, 2), (1, 2)]
labels = [r"$q_1 \cdot q_2$", r"$q_1 \cdot q_3$", r"$q_2 \cdot q_3$"]

cgs_errors = []
mgs_errors = []

for i, j in pairs:
    cgs_errors.append(abs(I_cgs[i, j]))
    mgs_errors.append(abs(I_mgs[i, j]))

x = np.arange(len(pairs))  # 0,1,2
width = 0.35

plt.figure(figsize=(7, 5))
plt.semilogy()  # y-axis log scale

plt.bar(x - width/2, cgs_errors, width, label="CGS")
plt.bar(x + width/2, mgs_errors, width, label="MGS")

plt.xticks(x, labels)
plt.ylabel(r"$|q_i^T q_j|$ (off-diagonal)")
plt.title("Orthogonality Error: CGS vs MGS")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

fig_name = os.path.join(FIG_DIR, "gram_schmidt_orthogonality_cgs_mgs.png")
plt.tight_layout()
plt.savefig(fig_name, dpi=300)
plt.show()

print(f"Figure saved to: {fig_name}")

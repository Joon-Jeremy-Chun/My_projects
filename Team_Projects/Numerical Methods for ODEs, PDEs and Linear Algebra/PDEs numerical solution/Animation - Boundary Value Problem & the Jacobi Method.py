# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:56:05 2024

@author: joonc
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def jacobi(A, b, u0, tol, max_iter):
    D = np.diag(np.diag(A))
    R = A - D
    D_inv = np.diag(1 / np.diag(D))
    u_approx = u0
    error = np.linalg.norm(np.dot(A, u_approx) - b, np.inf)
    num_iter = 0
    approximations = [u0]

    while error > tol and num_iter < max_iter:
        u_approx = np.dot(D_inv, b - np.dot(R, u_approx))
        error = np.linalg.norm(np.dot(A, u_approx) - b, np.inf)
        approximations.append(u_approx)
        num_iter += 1

    return approximations, num_iter
#%%
# Define constants p, L, q, and mesh size h
p = 7*10**(-6)
L = 6  # Beam length in feet
q = 4*10**(-7)
h = 1/2  # Mesh size
n = int(L / h) - 1

b = np.full((n, ), -h**2 * q)

# Initial guess u0
u0 = np.ones(n)

# Error tolerance and maximum number of iterations
tol = 1e-8
max_iter = 1000

# All other code remains the same until solving the system
A = np.zeros((n, n))
np.fill_diagonal(A, 2 + h**2 * p)  # Main diagonal
np.fill_diagonal(A[:-1, 1:], -1)  # Super-diagonal
np.fill_diagonal(A[1:, :-1], -1)  # Sub-diagonal
# Solve the system using the Jacobi method and record all approximations
approximations, num_iter = jacobi(A, b, u0, tol, max_iter)

# Plotting and Animation
x = np.linspace(h, L-h, n)
fig, ax = plt.subplots()
line, = ax.plot(x, approximations[0])
ax.set_xlim(h, L-h)
ax.set_ylim(-2, 2)  # Set y-axis range from -2 to 2
#ax.set_ylim(min(u0) - 0.1, max(u0) + 0.1)
ax.set_xlabel('Position along the beam (x)')
ax.set_ylabel('Deflection')
ax.set_title('Beam Deflection Due to Uniform Load')
ax.grid(True)

def animate(i):
    line.set_ydata(approximations[i])  # update the data
    return line,

ani = FuncAnimation(fig, animate, frames=num_iter, interval=100, blit=True, repeat=False)

plt.show()

print(f'Number of Jacobi iterations: {num_iter}')

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from numba import njit

medium = 30
plt.rc('text', usetex=True)

if not os.path.exists("images/2d_diff"):
    os.makedirs("images/2d_diff")

if not os.path.exists("data/2d_diff"):
    os.makedirs("data/2d_diff")

data_File = os.path.join("data")
image_File = os.path.join("images/2d_diff")

L = 64
N = 256
dx = L / N
dt = 0.01
D = 1
maxIter = 10001
skip = 1000
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)
sigma = 10
c = np.exp(-((X - L / 2) ** 2 + (Y - L / 2) ** 2) / (2 * sigma ** 2))
@njit
def evolve(c):
    dc = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            inext = (i + 1) % N
            iprev = (i - 1) % N
            jnext = (j + 1) % N
            jprev = (j - 1) % N
            term_x = (D / dx**2) * (c[inext, j] + c[iprev, j] - 2 * c[i, j])
            term_y = (D / dx**2) * (c[i, jnext] + c[i, jprev] - 2 * c[i, j])
            dc[i, j] = dt * (term_x + term_y)
    return dc

for t in range(maxIter):
    dc = evolve(c)
    c += dc
    if t % skip == 0:
        print("Iteration=%d,Total Concentration=%.6f"%(t,np.sum(c)))
        cFile = os.path.join(data_File, f"c_{t}.txt")
        np.savetxt(cFile, c)
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        ax.imshow(c, extent=[0, L, 0, L], origin="lower", cmap="viridis")
        ax.set_xlabel(r"$x$", fontsize=medium)
        ax.set_ylabel(r"$y$", fontsize=medium)
        plt.tight_layout()
        fig.savefig(os.path.join(image_File, f"c_{t}.png"), dpi=150)
        plt.close(fig)


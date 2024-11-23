import numpy as np
import matplotlib.pyplot as plt
import os
import sys
#Fontsize
medium = 30
plt.rc('text',usetex=True)
if not os.path.exists("images/1d_diff"):
    os.makedirs("images/1d_diff")

if not os.path.exists("data/1d_diff"):
    os.makedirs("data/1d_diff")

data_File = os.path.join("data/1d_diff")
image_File = os.path.join("images/1d_diff")

L = 64
N = 256
dx = L / N
dt = 0.01
D = 1
maxIter = 10001
skip = 1000
x = np.linspace(0, L, N)
c = np.sin(x)

def evolve(c):
    dc = np.zeros(N)
    for i in range(N):
        inext = (i + 1) % N
        iprev = (i - 1) % N
        term = (D / (dx) ** 2) * (c[inext] + c[iprev] - 2 * c[i])
        dc[i] = dt * term
    return dc

for t in range(maxIter):
    dc = evolve(c)
    c += dc
    if t % skip == 0:
        print("Iteration=%d,Total_concentration=%.6f"%(t,np.sum(c)))
        cFile = os.path.join(data_File, f"c_{t}.txt")
        np.savetxt(cFile, c)
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        ax.plot(x, c)
        ax.set_xlabel(r"$x$",fontsize=medium)
        ax.set_ylabel(r"$c(x)$",fontsize=medium)
        ax.set_xlim(0,L)
        ax.set_ylim(-1,1)
        plt.tight_layout()
        fig.savefig(os.path.join(image_File, f"c_{t}.png"), dpi=150)
        plt.close(fig)


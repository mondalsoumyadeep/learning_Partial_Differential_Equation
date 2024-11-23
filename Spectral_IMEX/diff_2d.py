import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn, fftfreq
import os
import sys

medium = 30
plt.rc('text', usetex=True)

if not os.path.exists("images/2d_diff"):
    os.makedirs("images/2d_diff")

if not os.path.exists("data/2d_diff"):
    os.makedirs("data/2d_diff")

data_File = os.path.join("data/2d_diff")
image_File = os.path.join("images/2d_diff")

L = 64
N = 64
dx = L / N
dt = 0.01
D = 10
maxIter = int(1e4)
skip = 1000

kx = fftfreq(N, dx)
ky = fftfreq(N, dx)
kx, ky = np.meshgrid(kx, ky)
ksqr = kx**2 + ky**2

c = np.zeros((N, N))
center = N // 2
c[center, center] = 10
c_hat = fftn(c)

def evolve(c_hat, ksqr):
    term = -D * ksqr * c_hat
    return term

for t in range(1, maxIter+1):
    c_hat = c_hat + evolve(c_hat, ksqr) * dt
    c_real = np.real(ifftn(c_hat))
    if t % skip == 0:
        print("Iteration=%d, Total Concentration=%1.1f"%(t,np.sum(c_real)))
        cFile = os.path.join(data_File, f"c_{t}.txt")
        np.savetxt(cFile, c_real)
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        ax.imshow(c_real, extent=[0, L, 0, L], origin='lower', cmap='viridis')
        ax.set_xlabel(r'$x$', fontsize=medium)
        ax.set_ylabel(r'$y$', fontsize=medium)
        ax.set_title(f"Iteration =%d"%t, fontsize=medium)
        fig.colorbar(ax.imshow(c_real, extent=[0, L, 0, L], origin='lower', cmap='viridis'), ax=ax, label='Concentration')
        fig.savefig(os.path.join(image_File, f"c_{t}.png"), dpi=150)
        plt.close(fig)


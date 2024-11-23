import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn, fftfreq
import os
import sys

medium = 30
plt.rc('text', usetex=True)

L = 128
N = 256
dx = L / N
maxIter = 10000
skip = 1000
dt = 0.1
v = 2.0
mu = 1.0
kappa = 0.5
noise_str = 0.1

c0 = float(sys.argv[1])

kx = fftfreq(N, dx) * 2 * np.pi
ky = fftfreq(N, dx) * 2 * np.pi
kx, ky = np.meshgrid(kx, ky)
ksqr = kx**2 + ky**2

dealias_threshold = (2 * kx.max() / 3)**2

if not os.path.exists("images/2d_ch"):
    os.makedirs("images/2d_ch")

if not os.path.exists("data/2d_ch"):
    os.makedirs("data/2d_ch")

data_File = os.path.join("data/2d_ch")
image_File = os.path.join("images/2d_ch")

def initialize_field(c0):
    return c0 + noise_str * np.random.randn(N, N)

def non_lin(c):
    return 2 * v * c * (1 - c) * (1 - 2 * c)

def evolve_field(c_hat, non_lin_hat):
    mask = ksqr < dealias_threshold
    c_hat = mask * (c_hat - dt * mu * ksqr * non_lin_hat) / (1 + dt * mu * kappa * ksqr**2)
    return c_hat

c = initialize_field(c0)
c_hat = fftn(c)

fig, ax = plt.subplots(figsize=(8.5, 6.5))
im = ax.imshow(c, cmap='viridis', vmin=0.0, vmax=1.0, extent=(0, L, 0, L))
ax.set_title(r"Initial Concentration Field", fontsize=medium)
ax.set_xlabel(r"$x$", fontsize=medium)
ax.set_ylabel(r"$y$", fontsize=medium)
fig.colorbar(im, ax=ax, label=r"Concentration")
fig.savefig(os.path.join(image_File, f"c_0.png"), dpi=150)
plt.close(fig)

for t in range(1, maxIter + 1):
    non_lin_real = non_lin(c)
    non_lin_hat = fftn(non_lin_real)
    c_hat = evolve_field(c_hat, non_lin_hat)
    c = np.real(ifftn(c_hat))
    
    if t % skip == 0:
        print("Iteration=%d, Total Concentration=%1.1f"%(t,np.sum(c)))
        cFile = os.path.join(data_File, f"c_{t}.txt")
        np.savetxt(cFile, c)

        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        im = ax.imshow(c, cmap='viridis', vmin=0.0, vmax=1.0, extent=(0, L, 0, L))
        ax.set_title(rf"Time Step = {t}", fontsize=medium)
        ax.set_xlabel(r"$x$", fontsize=medium)
        ax.set_ylabel(r"$y$", fontsize=medium)
        fig.colorbar(im, ax=ax, label=r"$C$")
        fig.savefig(os.path.join(image_File, f"c_{t}.png"), dpi=150)
        plt.close(fig)


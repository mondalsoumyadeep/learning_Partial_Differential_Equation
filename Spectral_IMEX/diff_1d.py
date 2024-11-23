import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq
import os
import sys

medium = 30
plt.rc('text', usetex=True)

if not os.path.exists("images/1d_diff"):
    os.makedirs("images/1d_diff")

if not os.path.exists("data/1d_diff"):
    os.makedirs("data/1d_diff")

data_File = os.path.join("data/1d_diff")
image_File = os.path.join("images/1d_diff")

L = 64
N = 64
dx = L / N
D = 10
dt = 0.01
maxIter = 10000
skip = 1000

kx = 2 * np.pi * fftfreq(N, dx)
ksqr = kx**2

x = np.linspace(0, L, N)
c = np.sin(x)
c_k = fft(c)

plt.plot(x, c, label=f'Numerical t=0')
plt.xlim(0,L)
plt.ylim(-1,1)
plt.savefig(os.path.join(image_File, f"c_0.png"), dpi=250)
plt.xlabel(r'$x$', fontsize=medium)
plt.ylabel(r'$c(x)$', fontsize=medium)
plt.clf()

def evolve(c_k, ksqr):
    term = -D * ksqr * c_k * dt
    return term

for t in range(1, maxIter + 1):
    c_k = c_k + evolve(c_k, ksqr)
    c_analytical = c_k * np.exp(D * ksqr * dt)
    c_real = np.real(ifft(c_k))
    c_analytical_real = np.real(ifft(c_analytical))
    if t % skip == 0:
        print("Iteration=%d, Total concentration=%.6f" % (t, np.sum(c_real)))
        cFile = os.path.join(data_File, f"c_{t}.txt")
        np.savetxt(cFile, c_real)
        plt.plot(x, c_real, label=f'Numerical t={t}')
        plt.plot(x, c_analytical_real, label=f'Analytical t={t}', linestyle=':')
        plt.legend()
        plt.xlabel(r'$x$', fontsize=medium)
        plt.ylabel(r'$c(x)$', fontsize=medium)
        plt.xlim(0,L)
        plt.ylim(-1,1)
        plt.title(f"$t = t$", fontsize=medium)
        plt.tight_layout()
        plt.savefig(os.path.join(image_File, f"c_{t}.png"), dpi=250)
        plt.clf()


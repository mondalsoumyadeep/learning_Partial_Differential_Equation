import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn, fftfreq
import sys
import os

# Parameters
L = 128  # Domain size
N = 128  # Number of grid points
dx = L / N

kx = 2 * np.pi * fftfreq(N, dx)
ky = 2 * np.pi * fftfreq(N, dx)

kx, ky = np.meshgrid(kx, ky)
ksqr = kx**2 + ky**2

# Dealiasing mask using 2/3 rule
cutoff = (2/3) * kx.max()
dealias_mask = (np.abs(kx) <= cutoff) & (np.abs(ky) <= cutoff)

# Physical parameters
M = 1.0
W = 2.0
kappa = 0.5
dt = 0.1
maxIter = 1000001
phi_p_init = float(sys.argv[1])  # Initial condition from command-line argument
noise_str = 0.1
# Random seed for reproducibility
np.random.seed(42)

# Initial concentration field
phi = phi_p_init + noise_str * np.random.randn(N, N)

# Precomputed terms for ETDRK2 scheme
term = -kappa * M * ksqr**2
exp_term = np.exp(term * dt)

# Non-linear term function
def calculate_non_linear_term(phi):
    return 2 * M * W * phi * (1 - phi) * (1 - 2 * phi)

# Set up output directory
output_dir = "images/cahn_hilliard"
os.makedirs(output_dir, exist_ok=True)

# ETDRK2 time-stepping loop
for t in range(maxIter):
    # Fourier transform of phi
    phi_hat = fftn(phi) * dealias_mask

    # Non-linear term in Fourier space
    non_linear_term = calculate_non_linear_term(phi)
    non_linear_term_hat = -ksqr * fftn(non_linear_term) * dealias_mask

    # ETDRK2 intermediate step: a_n_hat
    a_n_hat = phi_hat * exp_term + (non_linear_term_hat * (exp_term - 1) / (term + np.finfo(float).eps))

    # Compute F(a_n, tn + dt)
    a_n = ifftn(a_n_hat).real
    non_linear_a_n = calculate_non_linear_term(a_n)
    non_linear_a_n_hat = -ksqr * fftn(non_linear_a_n) * dealias_mask

    # Update in Fourier space
    F_n_hat = non_linear_term_hat
    F_a_n_hat = non_linear_a_n_hat
    u_n_plus_1_hat = a_n_hat + ((F_a_n_hat - F_n_hat) * (exp_term - 1 - dt * term) / (dt * term**2 + np.finfo(float).eps))

    # Apply de-aliasing and inverse transform
    u_n_plus_1_hat *= dealias_mask
    phi = ifftn(u_n_plus_1_hat).real

    # Visualization and output
    if t % 1000 == 0:
        print(f"Iteration {t}, Total Concentration: {np.sum(phi):.2f}")
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(phi, cmap='viridis', vmin=0.0, vmax=1.0, extent=(0, L, 0, L))
        ax.set_title(rf"Time Step = {t}", fontsize=14)
        ax.set_xlabel(r"$x$", fontsize=12)
        ax.set_ylabel(r"$y$", fontsize=12)
        fig.colorbar(im, ax=ax, label=r"$\phi$")
        fig.savefig(os.path.join(output_dir, f"c_{t:04d}.png"), dpi=150)
        plt.close(fig)

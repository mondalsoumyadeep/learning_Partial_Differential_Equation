import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn, fftfreq
import os

# Parameters
L = 128  # Domain size
N = 128  # Number of grid points
dx = L / N

# Fourier space setup
kx = 2 * np.pi * fftfreq(N, dx)
ky = 2 * np.pi * fftfreq(N, dx)
kx, ky = np.meshgrid(kx, ky)
ksqr = kx**2 + ky**2

# De-aliasing mask (2/3 rule)
cutoff = (2 / 3) * kx.max()
dealias_mask = (np.abs(kx) <= cutoff) & (np.abs(ky) <= cutoff)

# Physical parameters
M = 1.0  # Mobility
W = 2.0  # Free energy parameter
kappa = 0.5  # Gradient energy coefficient
dt = 0.1  # Time step
maxIter = 5000  # Number of iterations
noise_str = 0.1  # Noise strength

# Initial condition
np.random.seed(42)  # For reproducibility
phi = 0.5 + noise_str * np.random.randn(N, N)

# Precomputed terms for ETDRK2 scheme
term = -kappa * M * ksqr**2
exp_term = np.exp(term * dt)
coef_etdrk2_1 = (exp_term - 1) / (term + np.finfo(float).eps)
coef_etdrk2_2 = (
    exp_term - 1 - term * dt
) / (term**2 * dt + np.finfo(float).eps)

# Nonlinear term function
def calculate_non_linear_term(phi):
    return 2 * M * W * phi * (1 - phi) * (1 - 2 * phi)

# Create output directory
output_dir = "images/cahn_hilliard"
os.makedirs(output_dir, exist_ok=True)

# Time-stepping loop
for t in range(maxIter):
    # Fourier transform of phi
    phi_hat = fftn(phi) * dealias_mask

    # Nonlinear term in Fourier space
    non_linear_term = calculate_non_linear_term(phi)
    non_linear_term_hat = -ksqr * fftn(non_linear_term) * dealias_mask

    # Stage 1: Intermediate prediction step
    a_n_hat = phi_hat * exp_term + coef_etdrk2_1 * non_linear_term_hat
    a_n = ifftn(a_n_hat).real

    # Stage 2: Final correction
    non_linear_a_n = calculate_non_linear_term(a_n)
    non_linear_a_n_hat = -ksqr * fftn(non_linear_a_n) * dealias_mask
    phi_hat = a_n_hat + coef_etdrk2_2 * (non_linear_a_n_hat - non_linear_term_hat)

    # Apply de-aliasing and inverse transform
    phi_hat *= dealias_mask
    phi = ifftn(phi_hat).real

    # Visualization and output
    if t % 100 == 0:
        print(f"Iteration {t}, Total Concentration: {np.sum(phi):.2f}")
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(phi, cmap="viridis", vmin=0.0, vmax=1.0, extent=(0, L, 0, L))
        ax.set_title(f"Time Step = {t}", fontsize=14)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        fig.colorbar(im, ax=ax, label="$\phi$")
        fig.savefig(os.path.join(output_dir, f"c_{t:04d}.png"), dpi=150)
        plt.close(fig)


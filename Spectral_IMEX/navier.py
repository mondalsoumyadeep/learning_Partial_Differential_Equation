import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn, fftfreq
import os

medium = 30

if not os.path.exists("images/navier"):
    os.makedirs("images/navier")
image_File = os.path.join("images/navier")
plt.rc('text',usetex=True)
L = 1
N = 512
maxIter = 2001
skip = 100 
dt = 1e-3 
nu = 0.001  
dx = L / N  
kx = fftfreq(N, dx) * 2 * np.pi
ky = fftfreq(N, dx) * 2 * np.pi
kx, ky = np.meshgrid(kx, ky)
k_sqr = kx**2 + ky**2
k_sqr_inv = np.zeros((N,N))
k_sqr_inv[k_sqr > 0] = 1.0 / k_sqr[k_sqr > 0]
dealias_threshold = (2 * kx.max() / 3) ** 2
xlin = np.linspace(0, L, N)
xx, yy = np.meshgrid(xlin, xlin)
vx = -np.sin(2 * np.pi * yy)
vy = np.sin(4* np.pi * xx)


def lap(v_hat, k_sqr):
    return -k_sqr * v_hat

def grad(v_hat, kx, ky):
    dvx = 1.0j * kx * v_hat
    dvy = 1.0j * ky * v_hat
    return dvx, dvy

def div(vx_hat, vy_hat, kx, ky):
    return 1.0j * kx * vx_hat + 1.0j * ky * vy_hat

def curl(vx_hat, vy_hat, kx, ky):
    return np.real(ifftn(1.0j * kx * vy_hat - 1.0j * ky * vx_hat))

def non_linear_terms(vx, vy, kx, ky):
    dvx_x_hat, dvx_y_hat = grad(fftn(vx), kx, ky)
    dvy_x_hat, dvy_y_hat = grad(fftn(vy), kx, ky)

    non_lin_x = -(vx * np.real(ifftn(dvx_x_hat)) + vy * np.real(ifftn(dvx_y_hat)))
    non_lin_y = -(vx * np.real(ifftn(dvy_x_hat)) + vy * np.real(ifftn(dvy_y_hat)))
    
    return fftn(non_lin_x), fftn(non_lin_y)


def pressure_correction(vx_hat, vy_hat, kx, ky, k_sqr_inv):
    div_v_hat = div(vx_hat, vy_hat, kx, ky)
    P_hat = - div_v_hat * k_sqr_inv
    dP_x_hat, dP_y_hat = grad(P_hat, kx, ky)

    vx_hat -= dP_x_hat
    vy_hat -= dP_y_hat

    return vx_hat, vy_hat


def evolve_velocity(vx_hat, vy_hat, non_lin_x_hat, non_lin_y_hat, k_sqr, dt, nu, dealias_threshold):
    mask = k_sqr < dealias_threshold
    vx_hat = mask * (vx_hat + dt * non_lin_x_hat) / (1 + dt * nu * k_sqr)
    vy_hat = mask * (vy_hat + dt * non_lin_y_hat) / (1 + dt * nu * k_sqr)
    return vx_hat, vy_hat



for t in range(maxIter):
    vx_hat = fftn(vx)
    vy_hat = fftn(vy)
    
    non_lin_x_hat, non_lin_y_hat = non_linear_terms(vx, vy, kx, ky)
    vx_hat, vy_hat = pressure_correction(vx_hat, vy_hat, kx, ky, k_sqr_inv)
    vx_hat, vy_hat = evolve_velocity(vx_hat, vy_hat, non_lin_x_hat, non_lin_y_hat, k_sqr, dt, nu, dealias_threshold)

    vx = np.real(ifftn(vx_hat))
    vy = np.real(ifftn(vy_hat))
    
    if t % skip == 0:
        vorticity = curl(vx_hat, vy_hat, kx, ky)
        plt.imshow(vorticity,cmap='RdYlBu_r', extent=(0, L, 0, L))
        plt.title(f'Time step {t}')
        plt.colorbar()
        plt.savefig(os.path.join(image_File, f"c_{t}.png"), dpi=150)
        plt.clf()




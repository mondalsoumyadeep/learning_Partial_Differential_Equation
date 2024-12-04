import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftfreq,ifftn,fftn
L = 128
N = 128

dx = L/N

kx = 2*np.pi*fftfreq(N,dx)
ksqr = kx**2
D = 5.0
M = 1
dt = 0.01
maxiter = 1000

def non_linear(phi):
    return M*phi**2

phi = 0.5+0.1*np.random.rand(N)


for t in range(maxiter):
    phi_hat = fftn(phi)

    term = - D* ksqr  # Linear operator
    exp_term = np.exp(term * dt)  # e^{L dt}

    non_linear_term = non_linear(phi)
    a_n_hat = phi_hat * exp_term + (non_linear_term * (exp_term - 1)) / (term + 1e-10)  # Avoid division by zero

    # Inverse Transform to get a_n in real space
    a_n = ifftn(a_n_hat).real

    # Step 2: Calculate F(a_n, tn + h)
    non_linear_a_n = non_linear(a_n)

    # Step 3: Calculate the final update
    F_n = non_linear_term
    F_a_n = non_linear_a_n
    u_n_plus_1_hat = a_n_hat + (F_a_n - F_n) * ((exp_term - 1 - dt * term) / (dt**2 * term**2 + 1e-10))  
    
    # Inverse Transform to get updated concentration in real space
    phi_hat = u_n_plus_1_hat  
    phi = ifftn(phi_hat).real
    if t%500==0:
        print(np.sum(phi))
        plt.plot(phi)
        plt.ylim(0,1)
        plt.savefig("phi_%d.png"%t)





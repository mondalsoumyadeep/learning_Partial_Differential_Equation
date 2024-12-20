# Numerical Implementation of Partial Differential Equations


This repository contains codes for implementing nonlinear Partial Differential Equations (PDEs). The accompanying PDF provides an overview of the numerical methods used to solve these nonlinear PDEs.

In the Euler folder, I have solved the 1D and 2D diffusion equations using the Euler method. For coupled non-linear equations, visit this [link](https://github.com/mondalsoumyadeep/2024_Soft_Matter_Coarsening_of_aster_defects_in_a_model_polar_active_matter).

In the Spectral folder, I have solved the 1D and 2D diffusion equations, as well as the 2D Cahn-Hilliard equation using the pseudo-spectral scheme, which includes nonlinear terms as well as higher order gradients. 

Note: In every code, I have used plt.rc("text", usetex=True) to enable LaTeX rendering in plots. This may cause an error if LaTeX is not installed on your system. If you don't have LaTeX installed, you can simply remove this line.

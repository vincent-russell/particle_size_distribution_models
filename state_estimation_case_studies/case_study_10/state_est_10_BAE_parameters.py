"""
Parameters for condensation, deposition, and nucleation rate estimation example
"""


#######################################################
# Modules:
from numpy import array, eye

# Local modules:
from basic_tools import gaussian, diameter_to_volume, volume_to_diameter
from evolution_models.tools import Fuchs_Brownian


#######################################################
# Parameters:

# Setup and plotting:
smoothing = False  # Set to True to compute fixed interval Kalman smoother estimates
plot_animations = True  # Set to True to plot animations
plot_nucleation = True  # Set to True to plot nucleation plot
plot_images = True  # Set to True to plot images
load_coagulation = True  # Set to True to load coagulation tensors

# Spatial domain:
Dp_min = 1  # Minimum diameter of particles (micro m)
Dp_max = 5  # Maximum diameter of particles (micro m)
vmin = diameter_to_volume(Dp_min)  # Minimum volume of particles (micro m^3)
vmax = diameter_to_volume(Dp_max)  # Maximum volume of particles (micro m^3)

# Time domain:
dt = (1 / 60) * 2  # Time step (hours)
T = 24  # End time (hours)
NT = int(T / dt)  # Total number of time steps

# Size distribution discretisation:
Ne = 32  # Number of elements
Np = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
N = Ne * Np  # Total degrees of freedom
# Size distribution discretisation for reduced model:
Ne_r = 8  # Number of elements (needs to be a multiple of Ne)
Np_r = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
N_r = Ne_r * Np_r  # Total degrees of freedom

# Condensation rate discretisation:
Ne_gamma = 8  # Number of elements
Np_gamma = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
N_gamma = Ne_gamma * Np_gamma  # Total degrees of freedom
# Condensation rate discretisation for reduced model:
Ne_gamma_r = 2  # Number of elements (needs to be a multiple of Ne_gamma)
Np_gamma_r = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
N_gamma_r = Ne_gamma_r * Np_gamma_r  # Total degrees of freedom

# Deposition rate discretisation:
Ne_eta = 8  # Number of elements
Np_eta = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
N_eta = Ne_eta * Np_eta  # Total degrees of freedom
# Deposition rate discretisation for reduced model:
Ne_eta_r = 8  # Number of elements (needs to be a multiple of Ne_eta)
Np_eta_r = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
N_eta_r = Ne_eta_r * Np_eta_r  # Total degrees of freedom

# Initial guess of process rate paramters:
I_cst_guess = 0.5  # Condensation parameter constant
d_cst_guess = 0.4  # Deposition parameter constant

# Condensation VAR(p) coefficients for model gamma_{t + 1} = A_1 gamma_t + ... + w_{gamma_t}:
gamma_p = 5
gamma_A1 = 0.2 * eye(N_gamma)
gamma_A2 = 0.2 * eye(N_gamma)
gamma_A3 = 0.2 * eye(N_gamma)
gamma_A4 = 0.2 * eye(N_gamma)
gamma_A5 = 0.2 * eye(N_gamma)
gamma_A6 = 0 * eye(N_gamma)
gamma_A = array([gamma_A1, gamma_A2, gamma_A3, gamma_A4, gamma_A5, gamma_A6])  # Tensor of VAR(p) coefficients

# Deposition VAR(p) coefficients for model eta_{t + 1} = A_1 eta_t + ... + w_{eta_t}:
eta_p = 5
eta_A1 = 0.2 * eye(N_eta)
eta_A2 = 0.2 * eye(N_eta)
eta_A3 = 0.2 * eye(N_eta)
eta_A4 = 0.2 * eye(N_eta)
eta_A5 = 0.2 * eye(N_eta)
eta_A6 = 0 * eye(N_eta)
eta_A = array([eta_A1, eta_A2, eta_A3, eta_A4, eta_A5, eta_A6])  # Tensor of VAR(p) coefficients

# Nucleation AR(p) coefficients for model J_{t + 1} = a_1 J_t + ... + w_{J_t}:
J_p = 3  # Order of AR model
J_a1 = 2.2
J_a2 = -1.4
J_a3 = 0.2
J_a4 = 0
J_a5 = 0
J_a6 = 0
J_a = array([J_a1, J_a2, J_a3, J_a4, J_a5, J_a6])  # Vector of AR(p) coefficients

# Initial guess of the size distribution n_0(v) = n(v, 0):
N_0 = 1e3  # Amplitude of initial condition gaussian
v_0 = 10  # Mean of initial condition gaussian
sigma_0 = 2.5  # Standard deviation of initial condition gaussian
def initial_guess_size_distribution(v):
    return gaussian(v, N_0, v_0, sigma_0)

# Initial guess of the condensation rate I_0(Dp) = I_Dp(Dp, 0):
def initial_guess_condensation_rate(_):
    return I_cst_guess

# Initial guess of the deposition rate d_0(Dp) = d(Dp, 0):
def initial_guess_deposition_rate(_):
    return d_cst_guess

# Set to True for imposing boundary condition n(vmin, t) = 0:
boundary_zero = True

# Coagulation model:
def coag(v_x, v_y):
    Dp_x = volume_to_diameter(v_x)  # Diameter of particle x (micro m)
    Dp_y = volume_to_diameter(v_y)  # Diameter of particle y (micro m)
    return Fuchs_Brownian(Dp_x, Dp_y)

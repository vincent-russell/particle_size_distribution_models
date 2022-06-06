"""
Parameters for size estimation example
"""


#######################################################
# Local modules:
from basic_tools import gaussian, diameter_to_volume, volume_to_diameter
from evolution_models.tools import Fuchs_Brownian


#######################################################
# Parameters:

# Setup and plotting:
smoothing = False  # Set to True to compute fixed interval Kalman smoother estimates
plot_animations = True  # Set to True to plot animations
plot_nucleation = False  # Set to True to plot nucleation plot
plot_images = False  # Set to True to plot images
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

# Guess of process rate paramters:
I_cst_guess = 0.5  # Condensation parameter constant
d_cst_guess = 0.4  # Deposition parameter constant

# Prior noise parameters:
# Prior covariance for alpha; Gamma_alpha_prior = sigma_alpha_prior^2 * I_N (Size distribution):
sigma_alpha_prior_0 = 5
sigma_alpha_prior_1 = sigma_alpha_prior_0 / 2
sigma_alpha_prior_2 = sigma_alpha_prior_1 / 4
sigma_alpha_prior_3 = 0
sigma_alpha_prior_4 = 0
sigma_alpha_prior_5 = 0
sigma_alpha_prior_6 = 0

# Model noise parameters:
# Observation noise covariance parameters:
sigma_v = 1000  # Additive noise
sigma_Y_multiplier = 100  # Noise multiplier proportional to Y
# Evolution noise covariance Gamma_alpha_w = sigma_alpha_w^2 * I_N (Size distribution):
sigma_alpha_w_0 = sigma_alpha_prior_0
sigma_alpha_w_1 = sigma_alpha_prior_1
sigma_alpha_w_2 = sigma_alpha_prior_2
sigma_alpha_w_3 = 0
sigma_alpha_w_4 = 0
sigma_alpha_w_5 = 0
sigma_alpha_w_6 = 0
sigma_alpha_w_correlation = 2

# Modifying first element covariance for alpha (size distribution):
alpha_first_element_multiplier = 10

# Initial guess of the size distribution n_0(v) = n(v, 0):
N_0 = 1e3  # Amplitude of initial condition gaussian
v_0 = 10  # Mean of initial condition gaussian
sigma_0 = 2.5  # Standard deviation of initial condition gaussian
def initial_guess_size_distribution(v):
    return gaussian(v, N_0, v_0, sigma_0)

# Guess of the condensation rate I(Dp):
I_0_guess = 0.05  # Condensation parameter constant
I_1_guess = 1  # Condensation parameter inverse quadratic
def guess_cond(Dp):
    return I_0_guess + I_1_guess / (Dp ** 2)

# Guess of the deposition rate d(Dp):
d_0_guess = 1  # Deposition parameter constant
d_1_guess = -0.6  # Deposition parameter linear
d_2_guess = -(d_1_guess / 2) / 3  # Deposition parameter quadratic
def guess_depo(Dp):
    return d_0_guess + d_1_guess * Dp + d_2_guess * Dp ** 2  # Quadratic model output

# Guess of the nucleation rate J(t):
N_s_guess = 2e3  # Amplitude of gaussian nucleation event
t_s_guess = 8  # Mean time of gaussian nucleation event
sigma_s_guess = 1.5  # Standard deviation time of gaussian nucleation event
def guess_sorc(t):  # Source (nucleation) at vmin
    return gaussian(t, N_s_guess, t_s_guess, sigma_s_guess)  # Gaussian source (nucleation event) model output

# Set to True for imposing boundary condition n(vmin, t) = 0:
boundary_zero = True

# True underlying condensation model I_Dp(Dp, t):
I_0 = 0.05  # Condensation parameter constant
I_1 = 1  # Condensation parameter inverse quadratic
def cond(Dp):
    return I_0 + I_1 / (Dp ** 2)

# True underlying deposition model d(Dp, t):
d_0 = 1  # Deposition parameter constant
d_1 = -0.6  # Deposition parameter linear
d_2 = -(d_1 / 2) / 3  # Deposition parameter quadratic
def depo(Dp):
    return d_0 + d_1 * Dp + d_2 * Dp ** 2  # Quadratic model output

# True underlying source (nucleation event) model:
N_s = 2e3  # Amplitude of gaussian nucleation event
t_s = 8  # Mean time of gaussian nucleation event
sigma_s = 1.5  # Standard deviation time of gaussian nucleation event
def sorc(t):  # Source (nucleation) at vmin
    return gaussian(t, N_s, t_s, sigma_s)  # Gaussian source (nucleation event) model output

# Coagulation model:
def coag(v_x, v_y):
    Dp_x = volume_to_diameter(v_x)  # Diameter of particle x (micro m)
    Dp_y = volume_to_diameter(v_y)  # Diameter of particle y (micro m)
    return Fuchs_Brownian(Dp_x, Dp_y)

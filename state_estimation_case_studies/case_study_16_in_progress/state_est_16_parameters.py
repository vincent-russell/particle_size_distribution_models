"""
Parameters for state estimation
"""


#######################################################
# Modules:
from numpy import log, exp

# Local modules:
from basic_tools import skewed_gaussian, diameter_to_volume, volume_to_diameter
from evolution_models.tools import Fuchs_Brownian


#######################################################
# Parameters:

# Setup and plotting:
smoothing = True  # Set to True to compute fixed interval Kalman smoother estimates
plot_animations = True  # Set to True to plot animations
plot_nucleation = True  # Set to True to plot nucleation plot
plot_images = False  # Set to True to plot images
load_coagulation = True  # Set to True to load coagulation tensors
coagulation_suffix = 'CSTAR'  # Suffix of saved coagulation tensors file

# Spatial domain:
Dp_min = 0.0146  # Minimum diameter of particles (micro m)
Dp_max = 0.6612  # Maximum diameter of particles (micro m)
vmin = diameter_to_volume(Dp_min)  # Minimum volume of particles (micro m^3)
vmax = diameter_to_volume(Dp_max)  # Maximum volume of particles (micro m^3)
xmin = log(vmin)  # Lower limit in log-size
xmax = log(vmax)  # Upper limit in log-size

# Time domain:
dt = 0.0396  # Time step (hours)
NT = 167  # Total number of time steps
T = dt * NT  # End time (hours)

# Size distribution discretisation:
Ne = 50  # Number of elements
Np = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
N = Ne * Np  # Total degrees of freedom

# Prior noise parameters:
# Prior covariance for alpha; Gamma_alpha_prior = sigma_alpha_prior^2 * I_N (Size distribution):
sigma_alpha_prior_0 = 1
sigma_alpha_prior_1 = sigma_alpha_prior_0 / 2
sigma_alpha_prior_2 = sigma_alpha_prior_1 / 4
sigma_alpha_prior_3 = 0
sigma_alpha_prior_4 = 0
sigma_alpha_prior_5 = 0
sigma_alpha_prior_6 = 0

# Model noise parameters:
# Observation noise covariance parameters:
sigma_v = 50  # Additive noise
sigma_Y_multiplier = 0  # Noise multiplier proportional to Y
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
alpha_first_element_multiplier = 1

# Initial guess of the size distribution n_0(x) = n(x, 0):
N_0 = 53  # Amplitude of initial condition gaussian
x_0 = log(diameter_to_volume(0.023))  # Mean of initial condition gaussian
sigma_0 = 3.5  # Standard deviation of initial condition gaussian
skewness = 3  # Skewness factor for initial condition gaussian
def initial_guess_size_distribution(x):
    return skewed_gaussian(x, N_0, x_0, sigma_0, skewness)

# Guess of the condensation rate I(Dp):
I_cst_guess = 0.002  # Condensation parameter constant
I_linear_guess = 0  # Condensation parameter linear
def guess_cond(Dp):
    return I_cst_guess + I_linear_guess * Dp

# Guess of the deposition rate d(Dp):
d_cst_guess = 0.3  # Deposition parameter constant
d_linear_guess = 0  # Deposition parameter linear
def guess_depo(Dp):
    return d_cst_guess + d_linear_guess * Dp

# Set to True for imposing boundary condition n(vmin, t) = 0:
boundary_zero = False

# True underlying condensation model I_Dp(Dp, t):
I_cst = 0.0005  # Condensation parameter constant
I_linear = 0.01  # Condensation parameter linear
def cond(Dp):
    return I_cst + I_linear * Dp

# True underlying deposition model d(Dp, t):
d_cst = 0.15  # Deposition parameter constant
d_linear = 0.9  # Deposition parameter linear
def depo(Dp):
    return d_cst + d_linear * Dp

# Coagulation model:
def coag(x, y):
    v_x = exp(x)  # Volume of particle x (micro m^3)
    v_y = exp(y)  # Volume of particle y (micro m^3)
    Dp_x = volume_to_diameter(v_x)  # Diameter of particle x (micro m)
    Dp_y = volume_to_diameter(v_y)  # Diameter of particle y (micro m)
    return Fuchs_Brownian(Dp_x, Dp_y)

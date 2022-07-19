"""
Parameters for state estimation
"""


#######################################################
# Modules:
from numpy import log, exp

# Local modules:
from basic_tools import gaussian, skewed_gaussian, diameter_to_volume, volume_to_diameter
from evolution_models.tools import Fuchs_Brownian


#######################################################
# Parameters:

# Setup and plotting:
use_BAE = True  # Set to True to use BAE
filename_BAE = 'state_est_12_BAE'  # Filename for BAE mean and covariance
compute_weighted_norm = True  # Set to True to compute weighted norm difference (weighted by inverse of sigma_n)
plot_norm_difference = True  # Set to True to plot norm difference between truth and estimates
smoothing = True  # Set to True to compute fixed interval Kalman smoother estimates
plot_animations = True  # Set to True to plot animations
plot_nucleation = True  # Set to True to plot nucleation plot
plot_images = True  # Set to True to plot images
load_coagulation = True  # Set to True to load coagulation tensors
coagulation_suffix = '0004_to_1_1_micro_metres'  # Suffix of saved coagulation tensors file
data_filename = 'observations_06'  # Filename for data of simulated observations

# Spatial domain:
Dp_min = 0.004  # Minimum diameter of particles (micro m)
Dp_max = 1.1  # Maximum diameter of particles (micro m)
vmin = diameter_to_volume(Dp_min)  # Minimum volume of particles (micro m^3)
vmax = diameter_to_volume(Dp_max)  # Maximum volume of particles (micro m^3)
xmin = log(vmin)  # Lower limit in log-size
xmax = log(vmax)  # Upper limit in log-size

# Time domain:
dt = (1 / 60) * 20  # Time step (hours)
T = 24  # End time (hours)
NT = int(T / dt)  # Total number of time steps

# Size distribution discretisation:
Ne = 25  # Number of elements
Np = 1  # Np - 1 = degree of Legendre polynomial approximation in each element
N = Ne * Np  # Total degrees of freedom

# Prior noise parameters:
# Prior covariance for alpha; Gamma_alpha_prior = sigma_alpha_prior^2 * I_N (Size distribution):
sigma_alpha_prior_0 = 25
sigma_alpha_prior_1 = sigma_alpha_prior_0 / 2
sigma_alpha_prior_2 = sigma_alpha_prior_1 / 4
sigma_alpha_prior_3 = 0
sigma_alpha_prior_4 = 0
sigma_alpha_prior_5 = 0
sigma_alpha_prior_6 = 0

# Model noise parameters:
# Observation noise covariance parameters:
sigma_v = 4000  # Additive noise
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
alpha_first_element_multiplier = 10

# Initial guess of the size distribution n_0(x) = n(x, 0):
N_0 = 1e3  # Amplitude of initial condition gaussian
x_0 = log(diameter_to_volume(0.01))  # Mean of initial condition gaussian
sigma_0 = 3  # Standard deviation of initial condition gaussian
skewness = 3  # Skewness factor for initial condition gaussian
def initial_guess_size_distribution(x):
    return skewed_gaussian(x, N_0, x_0, sigma_0, skewness)

# Guess of the condensation rate I(Dp):
I_cst_guess = 0.002  # Condensation parameter constant
I_linear_guess = 0  # Condensation parameter linear
def guess_cond(Dp):
    return I_cst_guess + I_linear_guess * Dp

# Guess of the deposition rate d(Dp):
d_cst_guess = 0.1  # Deposition parameter constant
d_linear_guess = 0 # Deposition parameter linear
d_inverse_quadratic_guess = 0  # Deposition parameter inverse quadratic
def guess_depo(Dp):
    return d_cst_guess + d_linear_guess * Dp + d_inverse_quadratic_guess * (1 / Dp ** 2)

# Guess of the source (nucleation event) model:
N_s_guess = 2.5e3  # Amplitude of gaussian nucleation event
t_s_guess = 10  # Mean time of gaussian nucleation event
sigma_s_guess = 1  # Standard deviation time of gaussian nucleation event
def guess_sorc(t):  # Source (nucleation) at xmin
    return gaussian(t, N_s_guess, t_s_guess, sigma_s_guess)  # Gaussian source (nucleation event) model output

# Set to True for imposing boundary condition n(vmin, t) = 0:
boundary_zero = True

# True underlying condensation model I_Dp(Dp, t):
I_cst = 0.002  # Condensation parameter constant
I_linear = 0.05  # Condensation parameter linear
def cond(Dp):
    return I_cst + I_linear * Dp

# True underlying deposition model d(Dp, t):
d_cst = 0.02  # Deposition parameter constant
d_linear = 0.05  # Deposition parameter linear
d_inverse_quadratic = 0.00001  # Deposition parameter inverse quadratic
def depo(Dp):
    return d_cst + d_linear * Dp + d_inverse_quadratic * (1 / Dp ** 2)

# True underlying source (nucleation event) model:
N_s = 2e3  # Amplitude of gaussian nucleation event
t_s = 8  # Mean time of gaussian nucleation event
sigma_s = 1.5  # Standard deviation time of gaussian nucleation event
def sorc(t):  # Source (nucleation) at xmin
    return gaussian(t, N_s, t_s, sigma_s)  # Gaussian source (nucleation event) model output

# Coagulation model:
def coag(x, y):
    v_x = exp(x)  # Volume of particle x (micro m^3)
    v_y = exp(y)  # Volume of particle y (micro m^3)
    Dp_x = volume_to_diameter(v_x)  # Diameter of particle x (micro m)
    Dp_y = volume_to_diameter(v_y)  # Diameter of particle y (micro m)
    return Fuchs_Brownian(Dp_x, Dp_y)

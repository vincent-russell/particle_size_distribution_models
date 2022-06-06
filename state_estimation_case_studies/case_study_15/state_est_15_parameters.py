"""
Parameters for condensation, deposition, and nucleation rate estimation in log-size example (comparing to CSTAR)
"""


#######################################################
# Modules:
from numpy import log, exp, array, eye

# Local modules:
from basic_tools import gaussian, skewed_gaussian, diameter_to_volume, volume_to_diameter
from evolution_models.tools import Fuchs_Brownian


#######################################################
# Parameters:

# Setup and plotting:
smoothing = False  # Set to True to compute fixed interval Kalman smoother estimates
plot_animations = True  # Set to True to plot animations
plot_nucleation = False  # Set to True to plot nucleation plot
plot_images = True  # Set to True to plot images
load_coagulation = True  # Set to True to load coagulation tensors
coagulation_suffix = 'case_02'  # Suffix of saved coagulation tensors file

# Spatial domain:
Dp_min = 0.0146  # Minimum diameter of particles (micro m)
Dp_max = 0.6612  # Maximum diameter of particles (micro m)
vmin = diameter_to_volume(Dp_min)  # Minimum volume of particles (micro m^3)
vmax = diameter_to_volume(Dp_max)  # Maximum volume of particles (micro m^3)
xmin = log(vmin)  # Lower limit in log-size
xmax = log(vmax)  # Upper limit in log-size

# Time domain:
dt = 0.0396  # Total number of time steps
NT = 167  # Total number of time steps
T = dt * NT  # End time (hours)

# Size distribution discretisation:
Ne = 50  # Number of elements
Np = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
N = Ne * Np  # Total degrees of freedom

# Condensation rate discretisation:
Ne_gamma = 2  # Number of elements
Np_gamma = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
N_gamma = Ne_gamma * Np_gamma  # Total degrees of freedom

# Deposition rate discretisation:
Ne_eta = 2  # Number of elements
Np_eta = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
N_eta = Ne_eta * Np_eta  # Total degrees of freedom

# Initial guess of process rate paramters:
I_cst_guess = 0.005  # Condensation parameter constant
d_cst_guess = 0.5  # Deposition parameter constant

# Prior noise parameters:
# Prior covariance for alpha; Gamma_alpha_prior = sigma_alpha_prior^2 * I_N (Size distribution):
sigma_alpha_prior_0 = 0.05
sigma_alpha_prior_1 = sigma_alpha_prior_0 / 2
sigma_alpha_prior_2 = sigma_alpha_prior_1 / 4
sigma_alpha_prior_3 = 0
sigma_alpha_prior_4 = 0
sigma_alpha_prior_5 = 0
sigma_alpha_prior_6 = 0
# Prior covariance for gamma; Gamma_gamma_prior = sigma_gamma_prior^2 * I_N_gamma (Condensation rate):
sigma_gamma_prior_0 = 0.0025
sigma_gamma_prior_1 = sigma_gamma_prior_0 / 2
sigma_gamma_prior_2 = sigma_gamma_prior_1 / 4
sigma_gamma_prior_3 = 0
sigma_gamma_prior_4 = 0
sigma_gamma_prior_5 = 0
sigma_gamma_prior_6 = 0
# Prior covariance for eta; Gamma_eta_prior = sigma_eta_prior^2 * I_N_eta (Deposition rate):
sigma_eta_prior_0 = 0.25
sigma_eta_prior_1 = sigma_eta_prior_0 / 2
sigma_eta_prior_2 = sigma_eta_prior_1 / 4
sigma_eta_prior_3 = 0
sigma_eta_prior_4 = 0
sigma_eta_prior_5 = 0
sigma_eta_prior_6 = 0
# Prior uncertainty for J (Nucleation rate):
sigma_J_prior = 1

# Model noise parameters:
# Observation noise covariance parameters:
sigma_v = 30  # Additive noise
sigma_Y_multiplier = 10  # Noise multiplier proportional to Y
# Evolution noise covariance Gamma_alpha_w = sigma_alpha_w^2 * I_N (Size distribution):
sigma_alpha_w_0 = sigma_alpha_prior_0
sigma_alpha_w_1 = sigma_alpha_prior_1
sigma_alpha_w_2 = sigma_alpha_prior_2
sigma_alpha_w_3 = 0
sigma_alpha_w_4 = 0
sigma_alpha_w_5 = 0
sigma_alpha_w_6 = 0
sigma_alpha_w_correlation = 2
# Evolution noise covariance Gamma_gamma_w = sigma_gamma_w^2 * I_N_gamma (Condensation rate):
sigma_gamma_w_0 = sigma_gamma_prior_0 / 1000
sigma_gamma_w_1 = sigma_gamma_prior_1 / 1000
sigma_gamma_w_2 = sigma_gamma_prior_2 / 1000
sigma_gamma_w_3 = 0
sigma_gamma_w_4 = 0
sigma_gamma_w_5 = 0
sigma_gamma_w_6 = 0
sigma_gamma_w_correlation = 2
# Evolution noise covariance Gamma_eta_w = sigma_eta_w^2 * I_N_eta (Deposition rate):
sigma_eta_w_0 = sigma_eta_prior_0 / 1000
sigma_eta_w_1 = sigma_eta_prior_1 / 1000
sigma_eta_w_2 = sigma_eta_prior_2 / 1000
sigma_eta_w_3 = 0
sigma_eta_w_4 = 0
sigma_eta_w_5 = 0
sigma_eta_w_6 = 0
sigma_eta_w_correlation = 2
# Evolution noise for J (Nucleation rate):
sigma_J_w = sigma_J_prior

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
J_p = 1  # Order of AR model
J_a1 = 1
J_a2 = 0
J_a3 = 0
J_a4 = 0
J_a5 = 0
J_a6 = 0
J_a = array([J_a1, J_a2, J_a3, J_a4, J_a5, J_a6])  # Vector of AR(p) coefficients

# Modifying first element covariance for alpha (size distribution):
alpha_first_element_multiplier = 1
gamma_first_element_multiplier = 1
eta_first_element_multiplier = 1

# Initial condition n_0(x) = n(x, 0):
N_0 = 53  # Amplitude of initial condition gaussian
x_0 = log(diameter_to_volume(0.023))  # Mean of initial condition gaussian
sigma_0 = 3.5  # Standard deviation of initial condition gaussian
skewness = 3  # Skewness factor for initial condition gaussian
def initial_guess_size_distribution(x):
    return skewed_gaussian(x, N_0, x_0, sigma_0, skewness)

# Initial guess of the condensation rate I_0(Dp) = I(Dp, 0):
def initial_guess_condensation_rate(_):
    return I_cst_guess

# Initial guess of the deposition rate d_0(Dp) = d(Dp, 0):
def initial_guess_deposition_rate(_):
    return d_cst_guess

# Set to True for imposing boundary condition n(xmin, t) = 0:
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

# True underlying source (nucleation event) model:
N_s = 0  # Amplitude of gaussian nucleation event
t_s = 1  # Mean time of gaussian nucleation event
sigma_s = 1   # Standard deviation time of gaussian nucleation event
def sorc(t):  # Source (nucleation) at xmin
    return gaussian(t, N_s, t_s, sigma_s)  # Gaussian source (nucleation event) model output

# Coagulation model:
def coag(x, y):
    v_x = exp(x)  # Volume of particle x (micro m^3)
    v_y = exp(y)  # Volume of particle y (micro m^3)
    Dp_x = volume_to_diameter(v_x)  # Diameter of particle x (micro m)
    Dp_y = volume_to_diameter(v_y)  # Diameter of particle y (micro m)
    return Fuchs_Brownian(Dp_x, Dp_y)

"""
Parameters for state identification
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
smoothing = True  # Set to True to compute fixed interval Kalman smoother estimates
plot_animations = True  # Set to True to plot animations
plot_nucleation = True  # Set to True to plot nucleation plot
plot_images = False  # Set to True to plot images
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
dt = (1 / 60) * 5  # Time step (hours)
T = 48  # End time (hours)
NT = int(T / dt)  # Total number of time steps

# Size distribution discretisation:
Ne = 50  # Number of elements
Np = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
N = Ne * Np  # Total degrees of freedom

# Condensation rate discretisation:
Ne_gamma = 2  # Number of elements
Np_gamma = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
N_gamma = Ne_gamma * Np_gamma  # Total degrees of freedom

# Deposition rate discretisation:
Ne_eta = 3  # Number of elements
Np_eta = 2  # Np - 1 = degree of Legendre polynomial approximation in each element
N_eta = Ne_eta * Np_eta  # Total degrees of freedom

# Prior noise parameters:
# Prior covariance for alpha; Gamma_alpha_prior = sigma_alpha_prior^2 * I_N (Size distribution):
sigma_alpha_prior_0 = 10
sigma_alpha_prior_1 = sigma_alpha_prior_0 / 2
sigma_alpha_prior_2 = sigma_alpha_prior_1 / 4
sigma_alpha_prior_3 = 0
sigma_alpha_prior_4 = 0
sigma_alpha_prior_5 = 0
sigma_alpha_prior_6 = 0
# Prior covariance for gamma; Gamma_gamma_prior = sigma_gamma_prior^2 * I_N_gamma (Condensation rate):
sigma_gamma_prior_0 = 0.02
sigma_gamma_prior_1 = sigma_gamma_prior_0 / 2
sigma_gamma_prior_2 = sigma_gamma_prior_1 / 4
sigma_gamma_prior_3 = 0
sigma_gamma_prior_4 = 0
sigma_gamma_prior_5 = 0
sigma_gamma_prior_6 = 0
# Prior covariance for eta; Gamma_eta_prior = sigma_eta_prior^2 * I_N_eta (Deposition rate):
sigma_eta_prior_0 = 0.1
sigma_eta_prior_1 = sigma_eta_prior_0 / 2
sigma_eta_prior_2 = sigma_eta_prior_1 / 4
sigma_eta_prior_3 = 0
sigma_eta_prior_4 = 0
sigma_eta_prior_5 = 0
sigma_eta_prior_6 = 0
# Prior uncertainty for J (Nucleation rate):
sigma_J_prior = 20

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
# Evolution noise covariance Gamma_gamma_w = sigma_gamma_w^2 * I_N_gamma (Condensation rate):
sigma_gamma_w_0 = sigma_gamma_prior_0 / 100
sigma_gamma_w_1 = sigma_gamma_prior_1 / 100
sigma_gamma_w_2 = sigma_gamma_prior_2 / 100
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
gamma_p = 1
gamma_A1 = 1 * eye(N_gamma)
gamma_A2 = 0 * eye(N_gamma)
gamma_A3 = 0 * eye(N_gamma)
gamma_A4 = 0 * eye(N_gamma)
gamma_A5 = 0 * eye(N_gamma)
gamma_A6 = 0 * eye(N_gamma)
gamma_A = array([gamma_A1, gamma_A2, gamma_A3, gamma_A4, gamma_A5, gamma_A6])  # Tensor of VAR(p) coefficients

# Deposition VAR(p) coefficients for model eta_{t + 1} = A_1 eta_t + ... + w_{eta_t}:
eta_p = 1
eta_A1 = 1 * eye(N_eta)
eta_A2 = 0 * eye(N_eta)
eta_A3 = 0 * eye(N_eta)
eta_A4 = 0 * eye(N_eta)
eta_A5 = 0 * eye(N_eta)
eta_A6 = 0 * eye(N_eta)
eta_A = array([eta_A1, eta_A2, eta_A3, eta_A4, eta_A5, eta_A6])  # Tensor of VAR(p) coefficients

# Nucleation AR(p) coefficients for model J_{t + 1} = a_1 J_t + ... + w_{J_t}:
J_p = 6  # Order of AR model
J_a1 = 1.648
J_a2 = 0.088
J_a3 = -0.551
J_a4 = -0.435
J_a5 = -0.084
J_a6 = 0.335
J_a = array([J_a1, J_a2, J_a3, J_a4, J_a5, J_a6])  # Vector of AR(p) coefficients

# Modifying first element covariance for alpha (size distribution):
alpha_first_element_multiplier = 1000
gamma_first_element_multiplier = 1
eta_first_element_multiplier = 1

# Option to use element multiplier in covariance matrices (covariance decreases as element increases):
alpha_use_element_multipler = False
gamma_use_element_multipler = False
eta_use_element_multipler = False

# Initial guess of the size distribution n_0(x) = n(x, 0):
N_0 = 2e3  # Amplitude of initial condition gaussian
x_0 = log(diameter_to_volume(0.01))  # Mean of initial condition gaussian
sigma_0 = 3  # Standard deviation of initial condition gaussian
skewness = 3  # Skewness factor for initial condition gaussian
def initial_guess_size_distribution(x):
    return skewed_gaussian(x, N_0, x_0, sigma_0, skewness)

# Initial guess of the condensation rate I_0(Dp) = I_Dp(Dp, 0):
I_cst_guess = 0.01  # Condensation parameter constant
I_linear_guess = 0  # Condensation parameter linear
def initial_guess_condensation_rate(Dp):
    return I_cst_guess + I_linear_guess * Dp

# Initial guess of the deposition rate d_0(Dp) = d(Dp, 0):
d_cst_guess = 0.2  # Deposition parameter constant
d_linear_guess = 0  # Deposition parameter linear
d_inverse_quadratic_guess = 0  # Deposition parameter inverse quadratic
def initial_guess_deposition_rate(Dp):
    return d_cst_guess + d_linear_guess * Dp + d_inverse_quadratic_guess * (1 / Dp ** 2)

# Set to True for imposing boundary condition n(xmin, t) = 0:
boundary_zero = True

# True underlying condensation model I_Dp(Dp, t):
I_cst = 0.001  # Condensation parameter constant
I_linear = 0.08  # Condensation parameter linear
def cond(Dp):
    return I_cst + I_linear * Dp

# True underlying deposition model d(Dp, t):
d_cst = 0.05  # Deposition parameter constant
d_linear = 0.05  # Deposition parameter linear
d_inverse_quadratic = 0.000002  # Deposition parameter inverse quadratic
def depo(Dp):
    return d_cst + d_linear * Dp + d_inverse_quadratic * (1 / Dp ** 2)

# True underlying source (nucleation event) model:
N_s = 1.5e3  # Amplitude of gaussian nucleation event
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

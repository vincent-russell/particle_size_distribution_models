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
compute_weighted_norm = True  # Set to True to compute weighted norm difference (weighted by inverse of sigma_n)
plot_norm_difference = False  # Set to True to plot norm difference between truth and estimates
smoothing = True  # Set to True to compute fixed interval Kalman smoother estimates
plot_animations = False  # Set to True to plot animations
plot_nucleation = False  # Set to True to plot nucleation plot
plot_images = False  # Set to True to plot images
load_coagulation = True  # Set to True to load coagulation tensors
coagulation_suffix = 'evol_07'  # Suffix of saved coagulation tensors file
data_filename = 'observations_09'  # Filename for data of simulated observations

# Spatial domain:
Dp_min = 0.01  # Minimum diameter of particles (micro m)
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
Ne = 10  # Number of elements
Np = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
N = Ne * Np  # Total degrees of freedom

# Condensation rate discretisation:
Ne_gamma = 3  # Number of elements
Np_gamma = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
N_gamma = Ne_gamma * Np_gamma  # Total degrees of freedom

# Deposition rate discretisation:
Ne_eta = 3  # Number of elements
Np_eta = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
N_eta = Ne_eta * Np_eta  # Total degrees of freedom

# DMPS observation parameters:
use_DMPS_observation_model = True  # Set to True to use DMPS observation model
plot_dma_transfer_functions = False  # Set to True to plot DMA transfer functions
N_channels = 50  # Number of channels in DMA
R_inner = 0.937  # Inner radius of DMA (cm)
R_outer = 1.961  # Outer radius of DMA (cm)
length = 44.369 # Length of DMA (cm)
Q_aerosol = 0.3  # Aerosol sample flow (L/min)
Q_sheath = 3  # Sheath flow (L/min)
efficiency = 0.1  # Efficiency of DMA (flat percentage applied to particles passing through DMA); ranges from 0 to 1
voltage_min = 6  # Minimum voltage of DMA
voltage_max = 11000  # Maximum voltage of DMA
cpc_inlet_flow = 0.3  # CPC inlet flow (L/min)
cpc_count_time = 2  # Counting time for CPC inlet flow (seconds)

# Prior noise parameters:
# Prior covariance for alpha; Gamma_alpha_prior = sigma_alpha_prior^2 * I_N (Size distribution):
sigma_alpha_prior_0 = 1
sigma_alpha_prior_1 = sigma_alpha_prior_0 / 2
sigma_alpha_prior_2 = sigma_alpha_prior_1 / 4
sigma_alpha_prior_3 = 0
sigma_alpha_prior_4 = 0
sigma_alpha_prior_5 = 0
sigma_alpha_prior_6 = 0
# Prior covariance for gamma; Gamma_gamma_prior = sigma_gamma_prior^2 * I_N_gamma (Condensation rate):
sigma_gamma_prior_0 = 0.002 / 1  # Divide by 3 when doing random walk for fair comparison to VAR(p) models
sigma_gamma_prior_1 = sigma_gamma_prior_0 / 2
sigma_gamma_prior_2 = sigma_gamma_prior_1 / 4
sigma_gamma_prior_3 = 0
sigma_gamma_prior_4 = 0
sigma_gamma_prior_5 = 0
sigma_gamma_prior_6 = 0
# Prior covariance for eta; Gamma_eta_prior = sigma_eta_prior^2 * I_N_eta (Deposition rate):
sigma_eta_prior_0 = 0.003 / 1  # Divide by 3 when doing random walk for fair comparison to VAR(p) models. # Divide by 0.5 when doing test for filtering vs smoothing and coagulation vs no coagulation
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
sigma_v = 20  # Additive noise
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
sigma_gamma_w_0 = sigma_gamma_prior_0
sigma_gamma_w_1 = sigma_gamma_prior_1
sigma_gamma_w_2 = sigma_gamma_prior_2
sigma_gamma_w_3 = 0
sigma_gamma_w_4 = 0
sigma_gamma_w_5 = 0
sigma_gamma_w_6 = 0
sigma_gamma_w_correlation = 2
# Evolution noise covariance Gamma_eta_w = sigma_eta_w^2 * I_N_eta (Deposition rate):
sigma_eta_w_0 = sigma_eta_prior_0
sigma_eta_w_1 = sigma_eta_prior_1
sigma_eta_w_2 = sigma_eta_prior_2
sigma_eta_w_3 = 0
sigma_eta_w_4 = 0
sigma_eta_w_5 = 0
sigma_eta_w_6 = 0
sigma_eta_w_correlation = 2
# Evolution noise for J (Nucleation rate):
sigma_J_w = sigma_J_prior

# Condensation VAR(p) coefficients for model gamma_{t + 1} = A_1 gamma_t + ... + w_{gamma_t}:
gamma_loading_coefficients = False
gamma_loading_name = 'gamma_6_coefficients'
# If not loading coefficients can define here:
gamma_p = 1
gamma_A1 = 1 * eye(N_gamma)
gamma_A2 = 0 * eye(N_gamma)
gamma_A3 = 0 * eye(N_gamma)
gamma_A4 = 0 * eye(N_gamma)
gamma_A5 = 0 * eye(N_gamma)
gamma_A6 = 0 * eye(N_gamma)

# Deposition VAR(p) coefficients for model eta_{t + 1} = A_1 eta_t + ... + w_{eta_t}:
eta_loading_coefficients = False
eta_loading_name = 'eta_6_coefficients'
# If not loading coefficients can define here:
eta_p = 1
eta_A1 = 1 * eye(N_eta)
eta_A2 = 0 * eye(N_eta)
eta_A3 = 0 * eye(N_eta)
eta_A4 = 0 * eye(N_eta)
eta_A5 = 0 * eye(N_eta)
eta_A6 = 0 * eye(N_eta)

# Nucleation AR(p) coefficients for model J_{t + 1} = a_1 J_t + ... + w_{J_t}:
J_p = 3  # Order of AR model
J_a1 = 0.8
J_a2 = 0.4
J_a3 = -0.2
J_a4 = -0.477
J_a5 = -0.232
J_a6 = 0.264
J_a = array([J_a1, J_a2, J_a3, J_a4, J_a5, J_a6])  # Vector of AR(p) coefficients

# Modifying first element covariance for alpha (size distribution):
alpha_first_element_multiplier = 10
gamma_first_element_multiplier = 1
eta_first_element_multiplier = 1

# Option to use element multiplier in covariance matrices (covariance decreases as element increases):
alpha_use_element_multipler = False
gamma_use_element_multipler = False
eta_use_element_multipler = False

# Initial guess of the size distribution n_0(x) = n(x, 0):
N_0 = 2e3  # Amplitude of initial condition gaussian
x_0 = log(diameter_to_volume(0.03))  # Mean of initial condition gaussian
sigma_0 = 3  # Standard deviation of initial condition gaussian
skewness = 3  # Skewness factor for initial condition gaussian
def initial_guess_size_distribution(x):
    return skewed_gaussian(x, N_0, x_0, sigma_0, skewness)

# Initial guess of the condensation rate I_0(Dp) = I_Dp(Dp, 0):
I_cst_guess = 0.005  # Condensation parameter constant
I_linear_guess = 0  # Condensation parameter linear
def initial_guess_condensation_rate(Dp):
    return I_cst_guess + I_linear_guess * Dp

# Initial guess of the deposition rate d_0(Dp) = d(Dp, 0):
d_cst_guess = 0.05  # Deposition parameter constant
d_linear_guess = 0  # Deposition parameter linear
d_inverse_quadratic_guess = 0  # Deposition parameter inverse quadratic
def initial_guess_deposition_rate(Dp):
    return d_cst_guess + d_linear_guess * Dp + d_inverse_quadratic_guess * (1 / Dp ** 2)

# Set to True for imposing boundary condition n(xmin, t) = 0:
boundary_zero = True

# True underlying condensation model I_Dp(Dp, t):
I_cst = 0.0075  # Condensation parameter constant
I_linear = 0.04  # Condensation parameter linear
def cond_time(t):
    # Constant multiplier:
    cond_t_cst_amp = 1  # Amplitude
    cond_t_cst_mean = 8  # Mean time
    cond_t_cst_sigma = 3  # Standard deviation time
    cond_t_cst_multiplier = gaussian(t, cond_t_cst_amp, cond_t_cst_mean, cond_t_cst_sigma)
    # Linear multiplier:
    cond_t_linear_amp = 1  # Amplitude
    cond_t_linear_mean = 36  # Mean time
    cond_t_linear_sigma = 18  # Standard deviation time
    cond_t_linear_multiplier = gaussian(t, cond_t_linear_amp, cond_t_linear_mean, cond_t_linear_sigma)
    def cond(Dp):
        return cond_t_cst_multiplier * I_cst + cond_t_linear_multiplier * I_linear * Dp
    return cond

# True underlying deposition model d(Dp, t):
d_cst = 0.1  # Deposition parameter constant
d_linear = 0.1  # Deposition parameter linear
d_inverse_quadratic = 0.00001  # Deposition parameter inverse quadratic
def depo_time(t):
    # Constant multiplier:
    depo_t_cst_amp = 1  # Amplitude
    depo_t_cst_mean = 18  # Mean time
    depo_t_cst_sigma = 8  # Standard deviation time
    depo_t_cst_multiplier = gaussian(t, depo_t_cst_amp, depo_t_cst_mean, depo_t_cst_sigma)
    # Linear multiplier:
    depo_t_linear_amp = 1  # Amplitude
    depo_t_linear_mean = 36  # Mean time
    depo_t_linear_sigma = 8  # Standard deviation time
    depo_t_linear_multiplier = gaussian(t, depo_t_linear_amp, depo_t_linear_mean, depo_t_linear_sigma)
    # Quadratic multiplier:
    depo_t_quad_amp = 1  # Amplitude
    depo_t_quad_mean = 8  # Mean time
    depo_t_quad_sigma = 3  # Standard deviation time
    depo_t_quad_multiplier = gaussian(t, depo_t_quad_amp, depo_t_quad_mean, depo_t_quad_sigma)
    def depo(Dp):
        return depo_t_cst_multiplier * d_cst + depo_t_linear_multiplier * d_linear * Dp + depo_t_quad_multiplier * d_inverse_quadratic * (1 / Dp ** 2)
    return depo

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

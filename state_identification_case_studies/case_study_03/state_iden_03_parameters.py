"""
Parameters for state identification
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
compute_weighted_norm = True  # Set to True to compute weighted norm difference (weighted by inverse of sigma_n)
plot_norm_difference = False  # Set to True to plot norm difference between truth and estimates
smoothing = True  # Set to True to compute fixed interval Kalman smoother estimates
plot_animations = False  # Set to True to plot animations
plot_nucleation = False  # Set to True to plot nucleation plot
plot_images = False  # Set to True to plot images
load_coagulation = True  # Set to True to load coagulation tensors
coagulation_suffix = '1_to_10_micro_metres_diameter_true'  # Suffix of saved coagulation tensors file
discretise_with_diameter = True  # Set to True to uniformally discretise with diameter instead of volume
data_filename = 'observations_05'  # Filename for data of simulated observations

# Spatial domain:
Dp_min = 1  # Minimum diameter of particles (micro m)
Dp_max = 10  # Maximum diameter of particles (micro m)
vmin = diameter_to_volume(Dp_min)  # Minimum volume of particles (micro m^3)
vmax = diameter_to_volume(Dp_max)  # Maximum volume of particles (micro m^3)

# Time domain:
dt = (1 / 60) * 20  # Time step (hours)
T = 24  # End time (hours)
NT = int(T / dt)  # Total number of time steps

# Size distribution discretisation:
Ne = 50  # Number of elements
Np = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
N = Ne * Np  # Total degrees of freedom

# Condensation rate discretisation:
Ne_gamma = 10  # Number of elements
Np_gamma = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
N_gamma = Ne_gamma * Np_gamma  # Total degrees of freedom

# Deposition rate discretisation:
Ne_eta = 10  # Number of elements
Np_eta = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
N_eta = Ne_eta * Np_eta  # Total degrees of freedom

# DMPS observation parameters:
use_DMPS_observation_model = True  # Set to True to use DMPS observation model
plot_dma_transfer_functions = False  # Set to True to plot DMA transfer functions
N_channels = 20  # Number of channels in DMA
R_inner = 0.937  # Inner radius of DMA (cm)
R_outer = 1.961  # Outer radius of DMA (cm)
length = 44.369 # Length of DMA (cm)
Q_aerosol = 0.3  # Aerosol sample flow (L/min)
Q_sheath = 3  # Sheath flow (L/min)
efficiency = 0.08  # Efficiency of DMA (flat percentage applied to particles passing through DMA); ranges from 0 to 1
voltage_min = 12000  # Minimum voltage of DMA
voltage_max = 130000  # Maximum voltage of DMA
cpc_inlet_flow = 0.3  # CPC inlet flow (L/min)
cpc_count_time = 0.2  # Counting time for CPC inlet flow (seconds)

# Prior noise parameters:
# Prior covariance for alpha; Gamma_alpha_prior = sigma_alpha_prior^2 * I_N (Size distribution):
sigma_alpha_prior_0 = 5
sigma_alpha_prior_1 = sigma_alpha_prior_0 / 2
sigma_alpha_prior_2 = sigma_alpha_prior_1 / 4
sigma_alpha_prior_3 = 0
sigma_alpha_prior_4 = 0
sigma_alpha_prior_5 = 0
sigma_alpha_prior_6 = 0
# Prior covariance for gamma; Gamma_gamma_prior = sigma_gamma_prior^2 * I_N_gamma (Condensation rate):
sigma_gamma_prior_0 = 0.001
sigma_gamma_prior_1 = sigma_gamma_prior_0 / 2
sigma_gamma_prior_2 = sigma_gamma_prior_1 / 4
sigma_gamma_prior_3 = 0
sigma_gamma_prior_4 = 0
sigma_gamma_prior_5 = 0
sigma_gamma_prior_6 = 0
# Prior covariance for eta; Gamma_eta_prior = sigma_eta_prior^2 * I_N_eta (Deposition rate):
sigma_eta_prior_0 = 0.001
sigma_eta_prior_1 = sigma_eta_prior_0 / 2
sigma_eta_prior_2 = sigma_eta_prior_1 / 4
sigma_eta_prior_3 = 0
sigma_eta_prior_4 = 0
sigma_eta_prior_5 = 0
sigma_eta_prior_6 = 0
# Prior uncertainty for J (Nucleation rate):
sigma_J_prior = 400

# Model noise parameters:
# Observation noise covariance parameters:
sigma_v = 2000  # Additive noise
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
sigma_eta_w_1 = sigma_eta_prior_0 / 1000
sigma_eta_w_2 = sigma_eta_prior_0 / 1000
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
J_p = 1  # Order of AR model
J_a1 = 1
J_a2 = 0
J_a3 = 0
J_a4 = 0
J_a5 = 0
J_a6 = 0
J_a = array([J_a1, J_a2, J_a3, J_a4, J_a5, J_a6])  # Vector of AR(p) coefficients

# Modifying first element covariance for alpha (size distribution):
alpha_first_element_multiplier = 10
gamma_first_element_multiplier = 1
eta_first_element_multiplier = 1

# Option to use element multiplier in covariance matrices (covariance decreases as element increases):
alpha_use_element_multipler = True
gamma_use_element_multipler = False
eta_use_element_multipler = False

# Initial guess of the size distribution n_0(v) = n(v, 0):
N_0 = 300  # Amplitude of initial condition gaussian
v_0 = diameter_to_volume(4)  # Mean of initial condition gaussian
sigma_0 = 15  # Standard deviation of initial condition gaussian
def initial_guess_size_distribution(v):
    return gaussian(v, N_0, v_0, sigma_0)

# Initial guess of the condensation rate I_0(Dp) = I_Dp(Dp, 0):
I_0_guess = 0.2  # Condensation parameter constant
I_1_guess = 1  # Condensation parameter inverse quadratic
def initial_guess_condensation_rate(Dp):
    return I_0_guess + I_1_guess / (Dp ** 2)

# Initial guess of the deposition rate d_0(Dp) = d(Dp, 0):
depo_Dpmin_guess = 5  # Deposition parameter; diameter at which minimum
d_0_guess = 0.4  # Deposition parameter constant
d_1_guess = -0.15  # Deposition parameter linear
d_2_guess = -d_1_guess / (2 * depo_Dpmin_guess)  # Deposition parameter quadratic
def initial_guess_deposition_rate(Dp):
    return d_0_guess + d_1_guess * Dp + d_2_guess * Dp ** 2

# Set to True for imposing boundary condition n(vmin, t) = 0:
boundary_zero = True

# True underlying condensation model I_Dp(Dp, t):
I_0 = 0.2  # Condensation parameter constant
I_1 = 1  # Condensation parameter inverse quadratic
def cond(Dp):
    return I_0 + I_1 / (Dp ** 2)

# True underlying deposition model d(Dp, t):
depo_Dpmin = 5  # Deposition parameter; diameter at which minimum
d_0 = 0.4  # Deposition parameter constant
d_1 = -0.15  # Deposition parameter linear
d_2 = -d_1 / (2 * depo_Dpmin)  # Deposition parameter quadratic
def depo(Dp):
    return d_0 + d_1 * Dp + d_2 * Dp ** 2  # Quadratic model output

# True underlying source (nucleation event) model:
N_s = 5e3  # Amplitude of gaussian nucleation event
t_s = 8  # Mean time of gaussian nucleation event
sigma_s = 2  # Standard deviation time of gaussian nucleation event
def sorc(t):  # Source (nucleation) at vmin
    return gaussian(t, N_s, t_s, sigma_s)  # Gaussian source (nucleation event) model output

# Coagulation model:
def coag(v_x, v_y):
    Dp_x = volume_to_diameter(v_x)  # Diameter of particle x (micro m)
    Dp_y = volume_to_diameter(v_y)  # Diameter of particle y (micro m)
    return Fuchs_Brownian(Dp_x, Dp_y)

"""
Parameters for observation simulation
"""


#######################################################
# Modules:
from numpy import log, exp, linspace

# Local modules:
from basic_tools import gaussian, skewed_gaussian, diameter_to_volume, volume_to_diameter
from evolution_models.tools import Fuchs_Brownian


#######################################################
# Parameters:

# Setup and plotting:
plot_animations = True  # Set to True to plot animations
plot_nucleation = True  # Set to True to plot nucleation plot
plot_images = True  # Set to True to plot images
load_coagulation = True  # Set to True to load coagulation tensors
save_coagulation = False  # Set to True to save coagulation tensors
coagulation_suffix = 'evol_07'  # Suffix of saved coagulation tensors file

# Spatial domain:
Dp_min = 0.01  # Minimum diameter of particles (micro m)
Dp_max = 1  # Maximum diameter of particles (micro m)
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

# Observation parameters:
M = 50  # Observation dimension size
logDp_obs = linspace(log(Dp_min), log(Dp_max), M)  # Log(Diameters) that observations are made
sample_volume = 0.0005  # Volume of sample used in counting, y = (1 / sample_volume) * Pois(sample_volume * n)
additive_noise_mean = 0  # Mean of additive noise
additive_noise_sigma = 2  # Standard deviation of additive noise

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

# Save data parameters:
data_filename = 'observations_09'  # Filename for data of simulated observations

# Initial condition n_0(x) = n(x, 0):
N_0 = 2e3  # Amplitude of initial condition gaussian
x_0 = log(diameter_to_volume(0.03))  # Mean of initial condition gaussian
sigma_0 = 3  # Standard deviation of initial condition gaussian
skewness = 3  # Skewness factor for initial condition gaussian
def initial_condition(x):
    return skewed_gaussian(x, N_0, x_0, sigma_0, skewness)

# Set to True for imposing boundary condition n(xmin, t) = 0:
boundary_zero = True

# Condensation model I_Dp(Dp, t):
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

# Deposition model d(Dp, t):
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

# Source (nucleation event) model:
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

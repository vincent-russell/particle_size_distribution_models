"""
Parameters for observation simulation
"""


#######################################################
# Modules:
from numpy import log, exp, linspace

# Local modules:
from basic_tools import skewed_gaussian, diameter_to_volume, volume_to_diameter
from evolution_models.tools import Fuchs_Brownian


#######################################################
# Parameters:

# Setup and plotting:
plot_animations = True  # Set to True to plot animations
plot_images = False  # Set to True to plot images
load_coagulation = True  # Set to True to load coagulation tensors
save_coagulation = False  # Set to True to save coagulation tensors
coagulation_suffix = '01_to_10_micro_metres'  # Suffix of saved coagulation tensors file

# Spatial domain:
Dp_min = 0.1  # Minimum diameter of particles (micro m)
Dp_max = 10  # Maximum diameter of particles (micro m)
vmin = diameter_to_volume(Dp_min)  # Minimum volume of particles (micro m^3)
vmax = diameter_to_volume(Dp_max)  # Maximum volume of particles (micro m^3)
xmin = log(vmin)  # Lower limit in log-size
xmax = log(vmax)  # Upper limit in log-size

# Time domain:
dt = (1 / 60) * (1 / 2)  # Time step (hours)
T = 1  # End time (hours)
NT = int(T / dt)  # Total number of time steps

# Size distribution discretisation:
Ne = 50  # Number of elements
Np = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
N = Ne * Np  # Total degrees of freedom

# Observation parameters:
M = 10  # Observation dimension size
logDp_obs = linspace(log(Dp_min), log(Dp_max), M)  # Log(Diameters) that observations are made
sample_volume = 0.001  # Volume of sample used in counting, y = (1 / sample_volume) * Pois(sample_volume * n)

# DMPS observation parameters:
use_DMPS_observation_model = True  # Set to True to use DMPS observation model
plot_dma_transfer_functions = False  # Set to True to plot DMA transfer functions
N_channels = 10  # Number of channels in DMA
R_inner = 0.937  # Inner radius of DMA (cm)
R_outer = 1.961  # Outer radius of DMA (cm)
length = 44.369 # Length of DMA (cm)
Q_aerosol = 0.3  # Aerosol sample flow (L/min)
Q_sheath = 3  # Sheath flow (L/min)
efficiency = 0.08  # Efficiency of DMA (flat percentage applied to particles passing through DMA); ranges from 0 to 1
voltage_min = 600  # Minimum voltage of DMA
voltage_max = 130000  # Maximum voltage of DMA
cpc_inlet_flow = 0.3  # CPC inlet flow (L/min)
cpc_count_time = 2  # Counting time for CPC inlet flow (seconds)

# Save data parameters:
data_filename = 'observations_07'  # Filename for data of simulated observations

# Initial condition n_0(x) = n(x, 0):
N_0 = 1.5e3  # Amplitude of initial condition gaussian
N_1 = 1e3  # Amplitude of initial condition gaussian
x_0 = log(diameter_to_volume(0.2))  # Mean of initial condition gaussian
x_1 = log(diameter_to_volume(2))  # Mean of initial condition gaussian
sigma_0 = 2  # Standard deviation of initial condition gaussian
sigma_1 = 2  # Standard deviation of initial condition gaussian
skewness_0 = 5  # Skewness factor for initial condition gaussian
skewness_1 = 1  # Skewness factor for initial condition gaussian
def initial_condition(x):
    return skewed_gaussian(x, N_0, x_0, sigma_0, skewness_0) + skewed_gaussian(x, N_1, x_1, sigma_1, skewness_1)

# Set to True for imposing boundary condition n(xmin, t) = 0:
boundary_zero = True

# Condensation model I_Dp(Dp, t):
I_cst = 0.2  # Condensation parameter constant
I_linear = 0.5  # Condensation parameter linear
def cond(Dp):
    return I_cst + I_linear * Dp

# Deposition model d(Dp, t):
d_cst = 0.1  # Deposition parameter constant
d_linear = 0.5  # Deposition parameter linear
d_inv_linear = 0.25  # Deposition parameter inverse linear
def depo(Dp):
    return d_cst + d_linear * Dp + d_inv_linear * (1 / Dp)

# Coagulation model:
def coag(x, y):
    v_x = exp(x)  # Volume of particle x (micro m^3)
    v_y = exp(y)  # Volume of particle y (micro m^3)
    Dp_x = volume_to_diameter(v_x)  # Diameter of particle x (micro m)
    Dp_y = volume_to_diameter(v_y)  # Diameter of particle y (micro m)
    return Fuchs_Brownian(Dp_x, Dp_y)

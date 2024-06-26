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
coagulation_suffix = 'CSTAR_Dpmin_0008_diameter_true'  # Suffix of saved coagulation tensors file
discretise_with_diameter = True  # Set to True to uniformally discretise with diameter instead of volume

# Spatial domain:
Dp_min = 0.008  # Minimum diameter of particles (micro m)
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

# Observation parameters:
M = 107  # Observation dimension size
logDp_obs = linspace(log(Dp_min), log(Dp_max), M)  # Log(Diameters) that observations are made
sample_volume = 0.3  # Volume of sample used in counting, y = (1 / sample_volume) * Pois(sample_volume * n)

# Save data parameters:
data_filename = 'observations_08'  # Filename for data of simulated observations

# Initial condition n_0(x) = n(x, 0):
N_0 = 53  # Amplitude of initial condition gaussian
x_0 = log(diameter_to_volume(0.023))  # Mean of initial condition gaussian
sigma_0 = 3.25  # Standard deviation of initial condition gaussian
skewness = 3  # Skewness factor for initial condition gaussian
def initial_condition(x):
    return skewed_gaussian(x, N_0, x_0, sigma_0, skewness)

# Set to True for imposing boundary condition n(xmin, t) = 0:
boundary_zero = True

# Condensation model I_Dp(Dp, t):
I_cst = 0.001  # Condensation parameter constant
I_linear = 0.02  # Condensation parameter linear
def cond(Dp):
    return I_cst + I_linear * Dp

# Deposition model d(Dp, t):
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

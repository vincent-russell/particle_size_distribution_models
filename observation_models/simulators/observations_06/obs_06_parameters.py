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
plot_nucleation = False  # Set to True to plot nucleation plot
plot_images = False  # Set to True to plot images
load_coagulation = True  # Set to True to load coagulation tensors
save_coagulation = False  # Set to True to save coagulation tensors
coagulation_suffix = '0004_to_1_micro_metres'  # Suffix of saved coagulation tensors file

# Spatial domain:
Dp_min = 0.004  # Minimum diameter of particles (micro m)
Dp_max = 1  # Maximum diameter of particles (micro m)
vmin = diameter_to_volume(Dp_min)  # Minimum volume of particles (micro m^3)
vmax = diameter_to_volume(Dp_max)  # Maximum volume of particles (micro m^3)
xmin = log(vmin)  # Lower limit in log-size
xmax = log(vmax)  # Upper limit in log-size

# Time domain:
dt = (1 / 60) * 20  # Time step (hours)
T = 24  # End time (hours)
NT = int(T / dt)  # Total number of time steps

# Size distribution discretisation:
Ne = 50  # Number of elements
Np = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
N = Ne * Np  # Total degrees of freedom

# Observation parameters:
M = 50  # Observation dimension size
logDp_obs = linspace(log(Dp_min), log(Dp_max), M)  # Log(Diameters) that observations are made
sample_volume = 0.001  # Volume of sample used in counting, y = (1 / sample_volume) * Pois(sample_volume * n)

# Save data parameters:
data_filename = 'observations_06'  # Filename for data of simulated observations

# Initial condition n_0(x) = n(x, 0):
N_0 = 1e3  # Amplitude of initial condition gaussian
x_0 = log(diameter_to_volume(0.01))  # Mean of initial condition gaussian
sigma_0 = 3  # Standard deviation of initial condition gaussian
skewness = 3  # Skewness factor for initial condition gaussian
def initial_condition(x):
    return skewed_gaussian(x, N_0, x_0, sigma_0, skewness)

# Set to True for imposing boundary condition n(xmin, t) = 0:
boundary_zero = True

# Condensation model I_Dp(Dp, t):
I_cst = 0.002  # Condensation parameter constant
I_linear = 0.05  # Condensation parameter linear
def cond(Dp):
    return I_cst + I_linear * Dp

# Deposition model d(Dp, t):
d_cst = 0.02  # Deposition parameter constant
d_linear = 0.05  # Deposition parameter linear
d_inverse_quadratic = 0.00001  # Deposition parameter inverse quadratic
def depo(Dp):
    return d_cst + d_linear * Dp + d_inverse_quadratic * (1 / Dp ** 2)

# Source (nucleation event) model:
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

"""
Parameters for observation simulation
"""


#######################################################
# Modules:
from numpy import linspace

# Local modules:
from basic_tools import gaussian, diameter_to_volume, volume_to_diameter
from evolution_models.tools import Fuchs_Brownian


#######################################################
# Parameters:

# Setup and plotting:
plot_animations = True  # Set to True to plot animations
plot_images = False  # Set to True to plot images
load_coagulation = True  # Set to True to load coagulation tensors
save_coagulation = False  # Set to True to save coagulation tensors
coagulation_suffix = '1_to_10_micro_metres'  # Suffix of saved coagulation tensors file

# Spatial domain:
Dp_min = 1  # Minimum diameter of particles (micro m)
Dp_max = 10  # Maximum diameter of particles (micro m)
vmin = diameter_to_volume(Dp_min)  # Minimum volume of particles (micro m^3)
vmax = diameter_to_volume(Dp_max)  # Maximum volume of particles (micro m^3)

# Time domain:
dt = (1 / 60) * 5  # Time step (hours)
T = 24  # End time (hours)
NT = int(T / dt)  # Total number of time steps

# Size distribution discretisation:
Ne = 50  # Number of elements
Np = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
N = Ne * Np  # Total degrees of freedom

# Observation parameters:
M = 50  # Observation dimension size
d_obs = linspace(Dp_min, Dp_max, M)  # Diameters that observations are made
sample_volume = 0.005  # Volume of sample used in counting, y = (1 / sample_volume) * Pois(sample_volume * n)

# Save data parameters:
data_filename = 'observations_02'  # Filename for data of simulated observations

# Initial condition n_0(v) = n(v, 0):
N_0 = 300  # Amplitude of initial condition gaussian
v_0 = diameter_to_volume(4)  # Mean of initial condition gaussian
sigma_0 = 15  # Standard deviation of initial condition gaussian
def initial_condition(v):
    return gaussian(v, N_0, v_0, sigma_0)

# Set to True for imposing boundary condition n(vmin, t) = 0:
boundary_zero = True

# Condensation model I_Dp(Dp, t):
I_0 = 0.2  # Condensation parameter constant
I_1 = 1  # Condensation parameter inverse quadratic
def cond(Dp):
    return I_0 + I_1 / (Dp ** 2)

# Deposition model d(Dp, t):
depo_Dpmin = 5  # Deposition parameter; diameter at which minimum
d_0 = 0.4  # Deposition parameter constant
d_1 = -0.15  # Deposition parameter linear
d_2 = -d_1 / (2 * depo_Dpmin)  # Deposition parameter quadratic
def depo(Dp):
    return d_0 + d_1 * Dp + d_2 * Dp ** 2

# Coagulation model:
def coag(v_x, v_y):
    Dp_x = volume_to_diameter(v_x)  # Diameter of particle x (micro m)
    Dp_y = volume_to_diameter(v_y)  # Diameter of particle y (micro m)
    return Fuchs_Brownian(Dp_x, Dp_y)

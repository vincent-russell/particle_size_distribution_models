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
plot_nucleation = False  # Set to True to plot nucleation plot
plot_images = False  # Set to True to plot images
load_coagulation = True  # Set to True to load coagulation tensors
save_coagulation = False  # Set to True to save coagulation tensors
coagulation_suffix = '1_to_10_micro_metres_diameter_true'  # Suffix of saved coagulation tensors file
discretise_with_diameter = True  # Set to True to uniformally discretise with diameter instead of volume

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
Ne = 9 * 6  # Number of elements
Np = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
N = Ne * Np  # Total degrees of freedom

# Observation parameters:
M = 20  # Observation dimension size
d_obs = linspace(Dp_min, Dp_max, M)  # Diameters that observations are made
sample_volume = 0.001  # Volume of sample used in counting, y = (1 / sample_volume) * Pois(sample_volume * n)

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

# Save data parameters:
data_filename = 'observations_05_new'  # Filename for data of simulated observations

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

# Source (nucleation event) model:
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

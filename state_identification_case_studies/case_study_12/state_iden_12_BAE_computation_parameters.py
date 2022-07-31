"""
Parameters for BAE computation for state identification
"""


#######################################################
# Local modules:
from basic_tools import gaussian, diameter_to_volume, volume_to_diameter
from evolution_models.tools import Fuchs_Brownian


#######################################################
# Parameters:

# Setup:
load_coagulation = True  # Set to True to load coagulation tensors
coagulation_suffix = '1_to_11_micro_metres'  # Suffix of saved coagulation tensors file

# Spatial domain:
Dp_min = 1  # Minimum diameter of particles (micro m)
Dp_max = 11  # Maximum diameter of particles (micro m)
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
# Size discretisation for reduced model:
Ne_r = 25  # Number of elements (needs to be a multiple of Ne)
Np_r = 1  # Np - 1 = degree of Legendre polynomial approximation in each element
N_r = Ne_r * Np_r  # Total degrees of freedom

# Deposition rate discretisation:
Ne_eta = 3  # Number of elements
Np_eta = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
N_eta = Ne_eta * Np_eta  # Total degrees of freedom

# Deposition VAR(p) coefficients for model eta_{t + 1} = A_1 eta_t + ... + w_{eta_t}:
eta_p = 1  # Order of VAR model

# Nucleation AR(p) coefficients for model J_{t + 1} = a_1 J_t + ... + w_{J_t}:
J_p = 1  # Order of AR model

# Loop parameters:
filename_BAE = 'state_iden_07_1_BAE'  # Filename for BAE mean and covariance
N_iterations = 1000  # Number of samples from prior to compute BAE

# Coagulation model:
def coag(v_x, v_y):
    Dp_x = volume_to_diameter(v_x)  # Diameter of particle x (micro m)
    Dp_y = volume_to_diameter(v_y)  # Diameter of particle y (micro m)
    return Fuchs_Brownian(Dp_x, Dp_y)

# Set to True for imposing boundary condition n(vmin, t) = 0:
boundary_zero = True


#=========================================================#
# NOTE: The following parameters are for the reduced model.
#=========================================================#

# Initial guess of the size distribution n_0(v) = n(v, 0):
N_0 = 300  # Amplitude of initial condition gaussian
v_0 = diameter_to_volume(4)  # Mean of initial condition gaussian
sigma_0 = 15  # Standard deviation of initial condition gaussian
def initial_guess_size_distribution(v):
    return gaussian(v, N_0, v_0, sigma_0)

# Guess of the condensation rate I(Dp):
I_0_guess = 0.6  # Condensation parameter constant
I_1_guess = 0  # Condensation parameter inverse quadratic
def guess_cond(Dp):
    return I_0_guess + I_1_guess / (Dp ** 2)

"""
Parameters for BAE computation for state identification
"""


#######################################################
# Modules:
from numpy import log, exp

# Local modules:
from basic_tools import skewed_gaussian, diameter_to_volume, volume_to_diameter
from evolution_models.tools import Fuchs_Brownian


#######################################################
# Parameters:

# Setup:
load_coagulation = True  # Set to True to load coagulation tensors
coagulation_suffix = '01_to_11_micro_metres'  # Suffix of saved coagulation tensors file

# Spatial domain:
Dp_min = 0.1  # Minimum diameter of particles (micro m)
Dp_max = 11  # Maximum diameter of particles (micro m)
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
# Size discretisation for reduced model:
Ne_r = 25  # Number of elements (needs to be a multiple of Ne)
Np_r = 1  # Np - 1 = degree of Legendre polynomial approximation in each element
N_r = Ne_r * Np_r  # Total degrees of freedom

# Deposition rate discretisation:
Ne_eta = 2  # Number of elements
Np_eta = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
N_eta = Ne_eta * Np_eta  # Total degrees of freedom

# Deposition VAR(p) coefficients for model eta_{t + 1} = A_1 eta_t + ... + w_{eta_t}:
eta_p = 1  # Order of VAR model

# Loop parameters:
filename_BAE = 'state_iden_12_BAE'  # Filename for BAE mean and covariance
N_iterations = 1000  # Number of samples from prior to compute BAE

# Coagulation model:
def coag(x, y):
    v_x = exp(x)  # Volume of particle x (micro m^3)
    v_y = exp(y)  # Volume of particle y (micro m^3)
    Dp_x = volume_to_diameter(v_x)  # Diameter of particle x (micro m)
    Dp_y = volume_to_diameter(v_y)  # Diameter of particle y (micro m)
    return Fuchs_Brownian(Dp_x, Dp_y)

# Set to True for imposing boundary condition n(vmin, t) = 0:
boundary_zero = True


#=========================================================#
# NOTE: The following parameters are for the reduced model.
#=========================================================#

# Initial guess of the size distribution n_0(x) = n(x, 0):
N_0 = 1.5e3  # Amplitude of initial condition gaussian
N_1 = 1e3  # Amplitude of initial condition gaussian
x_0 = log(diameter_to_volume(0.2))  # Mean of initial condition gaussian
x_1 = log(diameter_to_volume(2))  # Mean of initial condition gaussian
sigma_0 = 2  # Standard deviation of initial condition gaussian
sigma_1 = 2  # Standard deviation of initial condition gaussian
skewness_0 = 5  # Skewness factor for initial condition gaussian
skewness_1 = 1  # Skewness factor for initial condition gaussian
def initial_guess_size_distribution(x):
    return skewed_gaussian(x, N_0, x_0, sigma_0, skewness_0) + skewed_gaussian(x, N_1, x_1, sigma_1, skewness_1)

# Guess of the condensation rate I(Dp):
I_cst_guess = 1  # Condensation parameter constant
I_linear_guess = 0  # Condensation parameter linear
def guess_cond(Dp):
    return I_cst_guess + I_linear_guess * Dp

"""
Parameter functions
"""


#######################################################
# Modules:
from numpy import pi, sqrt, exp

# Local modules:
from basic_tools import diameter_to_volume


#######################################################
# Physical constants, standard atmospheric conditions, and assumed parameters:
kb = 1.38064852e-23  # Boltzmann constant (units m^2 kg s^-2 K^-1)
T = 298  # Room temperature, equal to 25 degrees Celcius (units K)
rho = 0.001 * 1e6  # Aerosol particle density (kg m^-3)
mean_free_path = 0.0686 * 1e-6  # Mean free path of air at 298 K and atmospheric pressure (units m)
mu = 1.83e-5  # Viscosity of air at 298 K and atmospheric pressure (units kg m^-1 s^-1)


#######################################################
# Cunningham correction factor (dimensionless):
A1 = 1.257
A2 = 0.4
A3 = 0.55
def compute_correction_factor(Dp):
    C_coef = A1 + A2 * exp((-A3 * Dp) / mean_free_path)  # Dimensionless
    C = 1 + ((2 * mean_free_path) / Dp) * C_coef  # Dimensionless
    return C


#######################################################
# Diffusivity computation (units m^2 s^-1):
def compute_diffusivity(Dp):
    C = compute_correction_factor(Dp)  # Dimensionless
    D_coef = (kb * T * C) / (3 * pi * mu)  # Units m^3 s^-1
    D = D_coef / Dp  # Units m^2 s^-1
    return D


#######################################################
# Fuchs form of the Brownian coagulation coefficient (precomputations):
def compute_c_factor(v):
    m = rho * v
    c = sqrt((8 * kb * T) / (pi * m))
    return c

def compute_ell_factor(D, c):
    ell = (8 * D) / (pi * c)
    return ell

def compute_g_factor(Dp, ell):
    g_coef = (Dp + ell) ** 3 - (Dp ** 2 + ell ** 2) ** (3 / 2)
    g = (sqrt(2) / (3 * Dp * ell)) * g_coef - Dp
    return g


#######################################################
# Fuchs form of the Brownian coagulation coefficient:
def Fuchs_Brownian(Dp_1, Dp_2):
    # Converting units to standard units:
    Dp_1 = Dp_1 * 1e-6  # Micro meters to meters
    Dp_2 = Dp_2 * 1e-6  # Micro meters to meters
    # Computing volumes:
    v_1 = diameter_to_volume(Dp_1)  # Units m^3
    v_2 = diameter_to_volume(Dp_2)  # Units m^3
    # Computing D1 and D2:
    D1 = compute_diffusivity(Dp_1)
    D2 = compute_diffusivity(Dp_2)
    # Computing c1 and c2:
    c1 = compute_c_factor(v_1)
    c2 = compute_c_factor(v_2)
    c12 = sqrt(c1 ** 2 + c2 ** 2)
    # Computing ell1 and ell2:
    ell_1 = compute_ell_factor(D1, c1)
    ell_2 = compute_ell_factor(D2, c2)
    # Computing g1 and g2:
    g1 = compute_g_factor(Dp_1, ell_1)
    g2 = compute_g_factor(Dp_2, ell_2)
    g12 = 2 * sqrt(g1 ** 2 + g2 ** 2)
    # Computing coefficient:
    beta = ((Dp_1 + Dp_2) / (Dp_1 + Dp_2 + g12) + (8 * (D1 + D2)) / (c12 * (Dp_1 + Dp_2))) ** -1
    # Output:
    output = 2 * pi * (D1 + D2) * (Dp_1 + Dp_2) * beta
    # Change output units from m^3 s^-1 to cm^3 s^-1:
    output = output * 1e6
    return output

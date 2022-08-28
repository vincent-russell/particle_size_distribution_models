"""
Differential/Scanning Mobility Particle Sizer (DMPS/SMPS) classes and functions
"""


#######################################################
# Modules:
from math import floor
from numpy import pi, exp, e, log, log10, sqrt, linspace, zeros
from scipy.special import erf

# Local modules:
from basic_tools import volume_to_diameter, GLnpt


#######################################################
# Physical constants, standard atmospheric conditions, and assumed parameters:
kb = 1.38064852e-23  # Boltzmann constant (units m^2 kg s^-2 K^-1)
T = 298  # Room temperature, equal to 25 degrees Celcius (units K)
rho = 2000  # Aerosol particle density (kg m^-3)
mean_free_path = 0.0686 * 1e-6  # Mean free path of air at 298 K and atmospheric pressure (units m)
mu = 1.83e-5  # Viscosity of air at 298 K and atmospheric pressure (units kg m^-1 s^-1)
e_constant = 1.602176634e-19  # Elementary charge (coulombs)
G = 4.25  # G constant from Stolzenburg transfer theory for diffusing DMA factor


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
# Epsilon function in diffusing transfer function:
def epsilon(x):
    return x * erf(x) + (1 / sqrt(pi)) * exp(-(x ** 2))


#######################################################
# Function to get Differential Mobility Analyzer (DMA) transfer function based on Stolzenburg (1988) transfer theory:
def get_DMA_transfer_function(R_inner, R_outer, length, Q_aerosol, Q_sheath, efficiency):
        # Unit conversions:
        R_inner = R_inner / 100  # cm to m
        R_outer = R_outer / 100  # cm to m
        length = length / 100  # cm to m
        Q_aerosol = Q_aerosol / 60000  # L/min to m^3/sec
        Q_sheath = Q_sheath / 60000  # L/min to m^3/sec
        # Pre-computations:
        radius_ratio = log(R_outer / R_inner)  # Log of radius ratio
        beta = Q_aerosol / Q_sheath  # beta coefficient
        # Function of DMA transfer function (dimensionless form):
        def DMA_transfer_function_diffusing(Dp, voltage, n_charges):
            Dp = Dp / 1e6  # micro m to m
            C = compute_correction_factor(Dp)  # Cunningham correction factor (dimensionless)
            B = C / (3 * pi * mu * Dp)  # Particle mobility
            Zp_star = Q_sheath / (2 * pi * voltage * length) * radius_ratio  # Centriod of transfer function
            sigma_star = sqrt((G * kb * T) / (n_charges * e_constant * voltage) * radius_ratio)  # Spread of transfer function
            Zp = n_charges * e_constant * B  # Electrical particle mobility for a given number of charges
            Zp_tilde = Zp / Zp_star  # Dimensionless centriod of transfer function
            sigma = sigma_star * sqrt(Zp_tilde)  # Dimensionless spread of transfer function
            return efficiency * (sigma / (sqrt(2) * beta)) * (epsilon((Zp_tilde - (1 + beta)) / (sqrt(2) * sigma)) +
                                                 epsilon((Zp_tilde - (1 - beta)) / (sqrt(2) * sigma)) -
                                                 2 * epsilon((Zp_tilde - 1) / (sqrt(2) * sigma)))
        return DMA_transfer_function_diffusing


#######################################################
# Computes operator for computing z(t) given alpha(t), where y(t) ~ (1 / V) * Poisson(V * z(t)) are DMPS/SMPS measurements (and V is volume of sample through CPC):
def compute_alpha_to_z_operator(F, DMA_transfer_function, N_channels, voltage_min, voltage_max):
    if F.scale_type == 'log':
        voltage = exp(linspace(log(voltage_min), log(voltage_max), N_channels))  # Voltages of DMA (V)
    else:
        voltage = linspace(voltage_min, voltage_max, N_channels)  # Voltages of DMA (V)
    H_alpha_z = zeros([N_channels, F.N])  # Initialising
    for j in range(N_channels):  # Iterating over channels
        for i in range(F.N):  # Iterating over basis functions
            ell_i = floor(i / F.Np)  # ell-th element for phi_i
            # Integrand of H:
            if F.scale_type == 'log':
                def H_integrand(x):
                    v = exp(x)
                    Dp = volume_to_diameter(v)
                    return (3 / log10(e)) * DMA_transfer_function(Dp, voltage[j], 1) * F.phi[i](x)
            else:
                def H_integrand(v):
                    Dp = volume_to_diameter(v)
                    return ((pi * Dp ** 2) / 2) * DMA_transfer_function(Dp, voltage[j], 1) * F.phi[i](v)
            # Computing elements of H:
            H_alpha_z[j, i] = GLnpt(H_integrand, F.x_boundaries[ell_i], F.x_boundaries[ell_i + 1], 10)
    return H_alpha_z


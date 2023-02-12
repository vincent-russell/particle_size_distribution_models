"""
Discretisation functions
"""


#######################################################
# Modules:
import numpy as np
from numpy.polynomial.legendre import leggauss

# Local modules:
from basic_tools import get_kwarg_value


#######################################################
# Uniform discretisation of x_boundaries into Ne elements, with element size h, and Gauss nodes in each element:
def get_discretisation(Ne, Np, xmin, xmax):
    Np_Gauss_nodes = leggauss(Np)[0]  # Np Gauss nodes
    h = (xmax - xmin) / Ne  # Size of elements
    # Computing boundary values of elements:
    x_boundaries = np.zeros(Ne + 1)  # Initialising
    x_boundaries[0] = xmin  # First boundary value
    for ell in range(1, Ne + 1):
        x_boundaries[ell] = x_boundaries[ell - 1] + h  # Computing boundary values
    # Computing Gauss nodes in each element:
    x_ell_m = (x_boundaries[0:-1] + x_boundaries[1:]) / 2  # Mid-point of each element
    x_Gauss = np.zeros(Ne * Np)  # Initialising discretisation
    for ell in range(Ne):
        x_Gauss[ell * Np: (ell + 1) * Np] = x_ell_m[ell] + h / 2 * Np_Gauss_nodes  # Gauss nodes in element
    return x_Gauss, x_boundaries, h


#######################################################
# Compute plotting discretisation:
def get_plotting_discretisation(alpha, Gamma_alpha, x_boundaries, h, phi, N, Ne, Np, scale_type, **kwargs):
    # Parameters:
    time_varying = get_kwarg_value(kwargs, 'time_varying', True)  # Check if time varying
    return_Gamma = get_kwarg_value(kwargs, 'return_Gamma', False)  # Set to True to return Gamma matrix (covariance matrix)
    x_plot = get_kwarg_value(kwargs, 'x_plot', None)  # Return custom discretisation
    # x_plot computation:
    if x_plot is None:
        x_plot = np.zeros(N)  # Initialising
        for ell in range(Ne):  # Iterating over elements
            h_normalised = h[ell] / (Np + 1)  # Step size divided by order size
            for j in range(Np):  # Iterating over degrees
                i = j + ell * Np  # Getting total
                x_plot[i] = x_boundaries[ell] + (j + 1) * h_normalised  # Computing points within element
        # Adding boundary points:
        x_plot = np.insert(x_plot, 0, x_boundaries[0])
        x_plot = np.append(x_plot, x_boundaries[-1])
    N_plot = len(x_plot)  # Total discretisation length
    # Constructing phi matrix such that n = matrix * alpha and Gamma_n = matrix * Gamma_alpha * matrix^T:
    phi_matrix = np.zeros([N_plot, N])  # Initialising phi matrix (transform matrix from alpha to n)
    for i in range(N_plot):  # Iterations over x_plot discretisation
        for j in range(N):  # Iterations over basis functions
            phi_matrix[i, j] = phi[j](x_plot[i])  # Computing elements
    # Computing size distribution basis:
    Gamma_n, sigma_n = None, None  # Initialising
    if return_Gamma:
        n_plot, Gamma_n = change_basis_operator(alpha, Gamma_alpha, phi_matrix, time_varying=time_varying)
    else:
        n_plot, sigma_n = change_basis_operator(alpha, Gamma_alpha, phi_matrix, time_varying=time_varying, return_sigma=True)
    # Volume transformation if in log-scale (nm^3):
    if scale_type == 'log':
        v_plot = np.exp(x_plot)
    else:
        v_plot = x_plot
    # Radius (nm):
    r_plot = (abs((3 * v_plot) / (4 * np.pi))) ** (1 / 3)
    # Diameter (nm):
    d_plot = 2 * r_plot
    if return_Gamma:
        return d_plot, v_plot, n_plot, Gamma_n
    else:
        return d_plot, v_plot, n_plot, sigma_n


#######################################################
# Operator to change basis and statistics from x to y (y = A * x):
def change_basis_operator(x, Gamma_x, A, **kwargs):
    time_varying = get_kwarg_value(kwargs, 'time_varying', False)  # Check if time varying
    return_sigma = get_kwarg_value(kwargs, 'return_sigma', False)  # Set to True to return standard deviation
    if time_varying:
        N, NT = np.shape(x)  # Extracting dimensions of x
        M = np.shape(A)[0]  # Dimensions of Y
        AT = np.transpose(A)  # Transpose of basis mapping
        y = np.zeros([M, NT])  # Initialising y
        Gamma_y = np.zeros([NT, M, M])  # Initialising Gamma_y
        for t in range(NT):
            y[:, t] = np.matmul(A, x[:, t])  # Computing y
            Gamma_y[t] = np.matmul(A, np.matmul(Gamma_x[t], AT))  # Computing Gamma_y
        if return_sigma:
            sigma_y = np.zeros([M, NT])  # Initialising sigma_y
            for t in range(NT):
                sigma_y[:, t] = np.sqrt(np.diag(Gamma_y[t]))  # Computing sigma_y
            return y, sigma_y
        else:
            return y, Gamma_y
    else:
        AT = np.transpose(A)  # Transpose of basis mapping
        y = np.matmul(A, x)  # Computing y
        Gamma_y = np.matmul(A, np.matmul(Gamma_x, AT))  # Computing Gamma_y
        if return_sigma:
            sigma_y = np.sqrt(np.diag(Gamma_y))  # Computing sigma_y
            return y, sigma_y
        else:
            return y, Gamma_y


#######################################################
# Change size distribution based on particle volume to particle diameter:
def change_basis_volume_to_diameter(n_v, Dp):
    N, NT = np.shape(n_v)  # Array dimensions
    n_Dp = np.zeros([N, NT])  # Initialising
    for t in range(NT):
        for i in range(N):
            n_Dp[i, t] = ((np.pi * (Dp[i] ** 2)) / 2) * n_v[i, t]
    return n_Dp


#######################################################
# Change size distribution based on ln (natural log) to linear (e.g. from x = ln(v) to v):
def change_basis_ln_to_linear(n_x, v):
    N, NT = np.shape(n_x)  # Array dimensions
    n_v = np.zeros([N, NT])  # Initialising
    for t in range(NT):
        for i in range(N):
            n_v[i, t] = (1 / v[i]) * n_x[i, t]
    return n_v


#######################################################
# Change size distribution based on linear to log_10 (e.g. from v to x = log_10(v)):
def change_basis_linear_to_log(n_v, v):
    cst = 1 / np.log10(np.e)  # Constant for change of basis
    N, NT = np.shape(n_v)  # Array dimensions
    n_x = np.zeros([N, NT])  # Initialising
    for t in range(NT):
        for i in range(N):
            n_x[i, t] = cst * v[i] * n_v[i, t]
    return n_x


#######################################################
# Change size distribution based on x = ln(v) to log_10(Dp):
def change_basis_x_to_logDp(n_x, v, Dp):
    n_v = change_basis_ln_to_linear(n_x, v)
    n_Dp = change_basis_volume_to_diameter(n_v, Dp)
    return change_basis_linear_to_log(n_Dp, Dp)


#######################################################
# Change condensation rate based on particle volume to particle diameter:
def change_basis_volume_to_diameter_cond(I_v, Dp):
    N, NT = np.shape(I_v)  # Array dimensions
    I_Dp = np.zeros([N, NT])  # Initialising
    for t in range(NT):
        for i in range(N):
            I_Dp[i, t] = (2 / (np.pi * (Dp[i] ** 2))) * I_v[i, t]
    return I_Dp


#######################################################
# Change condensation rate based on ln (natural log) to linear (e.g. from x = ln(v) to v):
def change_basis_ln_to_linear_cond(I_x, v):
    N, NT = np.shape(I_x)  # Array dimensions
    I_v = np.zeros([N, NT])  # Initialising
    for t in range(NT):
        for i in range(N):
            I_v[i, t] = v[i] * I_x[i, t]
    return I_v


#######################################################
# Change condensation rate based on linear to log_10 (e.g. from v to x = log_10(v)):
def change_basis_linear_to_log_cond(I_v, v):
    cst = np.log10(np.e)  # Constant for change of basis
    N, NT = np.shape(I_v)  # Array dimensions
    I_x = np.zeros([N, NT])  # Initialising
    for t in range(NT):
        for i in range(N):
            I_x[i, t] = cst * (1 / v[i]) * I_v[i, t]
    return I_x


#######################################################
# Change condensation rate based on x = ln(v) to Dp:
def change_basis_x_to_Dp_cond(I_x, v, Dp):
    I_v = change_basis_ln_to_linear_cond(I_x, v)
    return change_basis_volume_to_diameter_cond(I_v, Dp)


#######################################################
# Change nucleation rate based on particle volume to particle diameter:
def change_basis_volume_to_diameter_sorc(J_v, Dp_min):
    NT = len(J_v)  # Array dimensions
    J_Dp = np.zeros(NT)  # Initialising
    for t in range(NT):
        J_Dp[t] = ((np.pi * (Dp_min ** 2)) / 2) * J_v[t]
    return J_Dp


#######################################################
# Change nucleation rate based on ln (natural log) to linear (e.g. from x = ln(v) to v):
def change_basis_ln_to_linear_sorc(J_x, vmin):
    NT = len(J_x)  # Array dimensions
    J_v = np.zeros(NT)  # Initialising
    for t in range(NT):
        J_v[t] = (1 / vmin) * J_x[t]
    return J_v


#######################################################
# Change nucleation rate based on linear to log_10 (e.g. from v to x = log_10(v)):
def change_basis_linear_to_log_sorc(J_v, vmin):
    cst = 1 / np.log10(np.e)  # Constant for change of basis
    NT = len(J_v)  # Array dimensions
    J_x = np.zeros(NT)  # Initialising
    for t in range(NT):
        J_x[t] = cst * vmin * J_v[t]
    return J_x


#######################################################
# Change nucleation rate based on x = ln(v) to log_10(Dp):
def change_basis_x_to_logDp_sorc(J_x, vmin, Dp_min):
    I_v = change_basis_ln_to_linear_sorc(J_x, vmin)
    J_Dp = change_basis_volume_to_diameter_sorc(I_v, Dp_min)
    return change_basis_linear_to_log_sorc(J_Dp, Dp_min)

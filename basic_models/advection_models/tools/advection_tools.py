"""
Advection tools
"""


#######################################################
# Modules:
import numpy as np
from numpy.polynomial.polynomial import polyval
from math import floor

# Local modules:
from basic_tools import GLnpt, Legendre, polyderivative, get_kwarg_value
from evolution_models.tools import (get_element_matrix, Phi_ell_vector, change_basis_operator)


#######################################################
# Uniform discretisation of x_boundaries into Ne elements, with element size h:
def get_discretisation(Ne, xmin, xmax):
    h = (xmax - xmin) / Ne  # Size of elements
    # Computing boundary values of elements:
    x_boundaries = np.zeros(Ne + 1)  # Initialising
    x_boundaries[0] = xmin  # First boundary value
    for ell in range(1, Ne + 1):
        x_boundaries[ell] = x_boundaries[ell - 1] + h  # Computing boundary values
    return x_boundaries, h


#######################################################
# Class j-th basis function, corresponding to i-th node in ell-th element,
# where Phi(j).eval(x) = phi_j(x) = phi^ell_i(x):
class Phi:

    def __init__(self, j, Np, x_boundaries):
        ell = floor(j / Np)  # ell-th element
        self.i = j - ell * Np  # i-th node in element
        self.x_lim = [x_boundaries[ell], x_boundaries[ell + 1]]  # Boundary to be non-zero
        self.xmid_ell = (x_boundaries[ell] + x_boundaries[ell + 1]) / 2  # Mid-point of ell-th element
        self.h_ell = x_boundaries[ell + 1] - x_boundaries[ell]  # Size of ell-th element

    def eval(self, x):
        if self.x_lim[0] < x < self.x_lim[1]:  # Non-zero only in element
            x_shifted = 2 * (x - self.xmid_ell) / self.h_ell  # Shifted value from element to interval [-1, 1]
            output = Legendre(x_shifted, self.i)  # Evaluating Legendre function
        elif x == self.x_lim[0]:  # If x is at lower boundary
            if self.i % 2 == 0:  # i.e. if i-th basis function is even
                output = 1
            else:  # Else i-th basis function is odd
                output = -1
        elif x == self.x_lim[1]:  # If x is at upper boundary
            output = 1  # Always equal to 1 whether basis function is odd or even
        else:  # x is not in non-zero boundary limits
            output = 0
        return output


#######################################################
# Class derivative of j-th basis function, corresponding to i-th node in ell-th element,
# where DPhi(j).eval(x) = phi_j'(x) = phi^ell_i'(x):
class DPhi:

    def __init__(self, j, Np, x_boundaries, phi):
        ell = floor(j / Np)  # ell-th element
        h_ell = x_boundaries[ell + 1] - x_boundaries[ell]  # Size of ell-th element
        self.i = j - ell * Np  # i-th node in element
        self.x_lim = [x_boundaries[ell], x_boundaries[ell + 1]]  # Boundary to be non-zero
        self.endpoint_value = self.i * (self.i + 1) / h_ell  # Function value at endpoints
        self.c = polyderivative(phi[j], x_boundaries[ell], x_boundaries[ell + 1], self.i)  # Coefficients for polynomial derivative

    def eval(self, x):
        if self.x_lim[0] < x < self.x_lim[1]:  # Non-zero only in element
            output = polyval(x, self.c)  # Evaluate derivative
        elif x == self.x_lim[0]:  # If x is at lower boundary
            if self.i % 2 == 0:  # i.e. if i-th basis function is even
                output = -self.endpoint_value
            else:  # Else i-th basis function is odd
                output = self.endpoint_value
        elif x == self.x_lim[1]:  # If x is at upper boundary
            output = self.endpoint_value  # Always equal to same value whether basis function is odd or even
        else:  # x is not in non-zero boundary limits
            output = 0
        return output


#######################################################
# Computes basis function vector
# phi[j].eval(x) = phi_j(x):
def get_Legendre_basis(N, Np, x_boundaries):
    phi = np.array([])
    for j in range(N):
        phi = np.append(phi, Phi(j, Np, x_boundaries).eval)
    return phi


#######################################################
# Computes derivative of basis function vector
# dphi[j].eval(x) = phi_j'(x):
def get_Legendre_basis_derivative(N, Np, x_boundaries, phi):
    dphi = np.array([])
    for j in range(N):
        dphi = np.append(dphi, DPhi(j, Np, x_boundaries, phi).eval)
    return dphi


#######################################################
# Computes matrix M:
def compute_M(N, Np, h):
    M = np.zeros([N, N])  # Initialising
    for i in range(N):
        ell_i = floor(i / Np)  # ell-th element for phi_i
        degree_i = i - ell_i * Np  # Degree of polynomial i
        for j in range(N):
            ell_j = floor(j / Np)  # ell-th element for phi_j
            degree_j = j - ell_j * Np  # Degree of polynomial j
            if ell_i == ell_j:  # Non-zero if in same element
                if degree_i == degree_j:  # Orthogonality property
                    M[j, i] = (h / 2) * (2 / (2 * degree_i + 1))  # Computing entries
    return M


#######################################################
# Computes matrix Q:
def compute_Q(advection, N, Np, x_boundaries, phi, dphi):
    Q = np.zeros([N, N])  # Initialising
    for i in range(N):
        ell_i = floor(i / Np)  # ell-th element for phi_i
        degree_i = i - ell_i * Np  # Degree of polynomial i
        for j in range(N):
            ell_j = floor(j / Np)  # ell-th element for phi_j
            degree_j = j - ell_j * Np  # Degree of polynomial j

            # Integrand in Q:
            def Q_integrand(x):
                return advection(x) * phi[i](x) * dphi[j](x)

            if ell_i == ell_j:  # Non-zero if in same element
                GLorder = floor((degree_i + degree_j + 1) / 2) + 2  # Order of integration of Gauss-Legendre quadrature
                Q[j, i] = GLnpt(Q_integrand, x_boundaries[ell_i], x_boundaries[ell_i + 1], GLorder)  # Computing entries
    return Q


#######################################################
# Computes matrix R (already assumes boundary condition at xmin is zero):
def compute_R(advection, Ne, Np, N, x_boundaries, phi, M, Q):
    R = np.zeros([N, N])  # Initialising matrix R
    for ell in range(Ne):
        x_ell = x_boundaries[ell]  # x-values of element lower bound
        x_ell_plus_1 = x_boundaries[ell + 1]  # x-values of element upper bound
        inv_M_ell = np.linalg.inv(get_element_matrix(M, ell, Np))  # Getting M_ell^{-1} matrix
        Q_ell = get_element_matrix(Q, ell, Np)  # Getting Q_ell matrix
        phi_vector = Phi_ell_vector(phi, ell, Np).get  # Getting Phi_ell(x) vector function
        phi_vector_minus_1 = Phi_ell_vector(phi, max(ell - 1, 0), Np).get  # Getting Phi_{ell-1}(x) vector function
        # Computing R1_ell and R2_ell matrices:
        u1 = np.zeros([Np, 1])  # Initialising vector
        u2 = np.zeros([Np, 1])  # Initialising vector
        u3 = np.zeros([Np, 1])  # Initialising vector
        u1[:, 0] = phi_vector(x_ell_plus_1)  # Vector value
        u2[:, 0] = phi_vector(x_ell)  # Vector value
        u3[:, 0] = phi_vector_minus_1(x_ell)  # Vector value
        R1_ell = advection(x_ell_plus_1) * np.matmul(u1, np.transpose(u1))  # P1_ell computation
        R2_ell = advection(x_ell) * np.matmul(u2, np.transpose(u3))  # P2_ell computation
        # Computing tilde_R1_ell and tilde_R2_ell matrices:
        tilde_R1_ell = np.matmul(inv_M_ell, np.subtract(Q_ell, R1_ell))
        tilde_R2_ell = np.matmul(inv_M_ell, R2_ell)
        # Adding elements to matrix:
        if ell == 0:  # First element only adds tilde_R1_ell
            R[ell * Np: (ell + 1) * Np, ell * Np: (ell + 1) * Np] = tilde_R1_ell
        else:
            R[ell * Np: (ell + 1) * Np, ell * Np: (ell + 1) * Np] = tilde_R1_ell
            R[ell * Np: (ell + 1) * Np, (ell - 1) * Np: ell * Np] = tilde_R2_ell
    return R


#######################################################
# Computing matrix G:
def compute_G(N, N_r, Np, Np_r, phi, phi_r, x_boundaries):
    G = np.zeros([N_r, N])  # Initialising
    for i in range(N):
        ell_i = floor(i / Np)  # ell-th element for phi_i
        degree_i = i - ell_i * Np  # Degree of polynomial i
        for j in range(N_r):
            ell_j = floor(j / Np_r)  # ell-th element for phi^r_j
            degree_j = j - ell_j * Np_r  # Degree of polynomial j

            # Integrand in G:
            def G_integrand(x):
                return phi_r[j](x) * phi[i](x)

            GLorder = floor((degree_i + degree_j + 1) / 2) + 3  # Order of integration of Gauss-Legendre quadrature
            G[j, i] = GLnpt(G_integrand, x_boundaries[ell_i], x_boundaries[ell_i + 1], GLorder)  # Computing entries
    return G


#######################################################
# Computes basis coefficients(t) from function f(x, t) and Legendre basis functions
# at fixed time (based on Hilbert projection theorem):
def compute_coefficients(f, N, Np, phi, x_boundaries, h):
    coefficients = np.zeros(N)  # Initialising
    for j in range(N):
        ell = floor(j / Np)  # ell-th element
        i = j - ell * Np  # i-th node in element

        # Projection integrand:
        def integrand(x):
            return f(x) * phi[j](x)
        GLorder = floor((i + 1) / 2) + 3  # Order of integration of Gauss-Legendre quadrature of integrand
        integral_approx = GLnpt(integrand, x_boundaries[ell], x_boundaries[ell + 1], GLorder)  # Computing integral in projection
        # Normalisation and alpha assimilation:
        normalise_constant = (h / 2) * (2 / (2 * i + 1))
        coefficients[j] = integral_approx / normalise_constant
    return coefficients


#######################################################
# Class that returns approximation function f(x, t), corresponding to basis coefficients(t) at
# time t, where Approximation.eval(x) = f(x, t) given coefficients(t):
class Approximation:

    def __init__(self, coefficient, phi, Np, Ne, x_boundaries, h):
        self.coefficient = coefficient  # Basis coefficients
        self.phi = phi  # Basis functions
        self.Np = Np  # Number of basis functions in each element
        self.Ne = Ne  # Number of elements
        self.x_boundaries = x_boundaries  # Boundary values of elements
        self.h = h  # Element size

    def eval(self, x):
        x_element_condition = np.logical_and(self.x_boundaries <= x, x <= (self.x_boundaries + self.h))  # Condition to find x in which element interval [x^ell, x^{ell + 1}]
        ell = np.where(x_element_condition)[0][0]  # Finds element ell such that x in [x^ell, x^{ell + 1}]
        ell = min(ell, self.Ne - 1)  # Condition to make sure that ell < Ne
        output = 0  # Initialising sum
        for j in range(ell * self.Np, (ell + 1) * self.Np):
            output += self.coefficient[j] * self.phi[j](x)  # Computing coefficient_j(t) * phi_j(x)
        return output


#######################################################
# Compute plotting discretisation:
def get_plotting_discretisation(coefs, xmin, xmax, NT, phi, Np, Ne, x_boundaries, h, N_plot):
    x_plot = np.linspace(xmin, xmax, N_plot)  # Plotting discretisation in x
    n_plot = np.zeros([N_plot, NT])  # Initialising
    for t in range(NT):  # Iterating over time
        n_func_t = Approximation(coefs[:, t], phi, Np, Ne, x_boundaries, h)  # Retrieving function from alpha coefficients
        for i in range(N_plot):  # Iterating over x
            n_plot[i, t] = n_func_t.eval(x_plot[i])  # Evaluating n(x, t)
    return x_plot, n_plot


#######################################################
# Compute plotting discretisation:
def get_plotting_discretisation_with_uncertainty(coefs, Gamma_coefs, xmin, xmax, phi, N, N_plot, **kwargs):
    # Parameters:
    return_Gamma = get_kwarg_value(kwargs, 'return_Gamma', False)  # Set to True to return Gamma matrix (covariance matrix)
    # x_plot computation:
    x_plot = np.linspace(xmin, xmax, N_plot)  # Plotting discretisation in x
    # Constructing phi matrix such that n = matrix * alpha and Gamma_n = matrix * Gamma_alpha * matrix^T:
    phi_matrix = np.zeros([N_plot, N])  # Initialising phi matrix (transform matrix from alpha to n)
    for i in range(N_plot):  # Iterations over x_plot discretisation
        for j in range(N):  # Iterations over basis functions
            phi_matrix[i, j] = phi[j](x_plot[i])  # Computing elements
    # Computing plotting discretisation:
    if return_Gamma:
        n_plot, Gamma_n = change_basis_operator(coefs, Gamma_coefs, phi_matrix, time_varying=True)
        return x_plot, n_plot, Gamma_n
    else:
        n_plot, sigma_n = change_basis_operator(coefs, Gamma_coefs, phi_matrix, time_varying=True, return_sigma=True)
        return x_plot, n_plot, sigma_n


#######################################################
# Observation model:
def compute_H(N, phi, x_obs, obs_dim):
    # Computing H matrix:
    H = np.zeros([obs_dim, N])  # Initialising observation matrix H
    for j in range(N):
        for i in range(obs_dim):
            H[i, j] = phi[j](x_obs[i])  # Computing elements of matrix H
    return H

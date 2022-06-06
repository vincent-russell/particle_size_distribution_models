"""
Legendre basis functions
"""


#######################################################
# Modules:
import numpy as np
from numpy.polynomial.polynomial import polyval
from math import floor

# Local modules:
from basic_tools.Legendre_polynomial_functions import Legendre
from basic_tools.polynomial_approximators import polyderivative
from basic_tools.numerical_integrators import GLnpt


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
# Computes basis function vector
# phi[j].eval(x) = phi_j(x):
def get_Legendre_basis(N, Np, x_boundaries):
    phi = np.array([])
    for j in range(N):
        phi = np.append(phi, Phi(j, Np, x_boundaries).eval)
    return phi


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
# Computes derivative of basis function vector
# dphi[j].eval(x) = phi_j'(x):
def get_Legendre_basis_derivative(N, Np, x_boundaries, phi):
    dphi = np.array([])
    for j in range(N):
        dphi = np.append(dphi, DPhi(j, Np, x_boundaries, phi).eval)
    return dphi


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
# Class that returns ell-th approximation function f(x, t), corresponding to basis coefficient^ell(t) at
# time t, where Approximation_ell.eval(x) = f^ell(x, t) given coefficient^ell(t):
class Approximation_ell:

    def __init__(self, coefficient_ell, phi_ell, Np):
        self.coefficient_ell = coefficient_ell  # Basis coefficients
        self.phi_ell = phi_ell  # Basis functions
        self.Np = Np  # Number of basis functions in each element

    def eval(self, x):
        output = 0  # Initialising sum
        for i in range(self.Np):
            output += self.coefficient_ell[i] * self.phi_ell[i](x)  # Computing coefficient^ell_i(t) * phi^ell_i(x)
        return output

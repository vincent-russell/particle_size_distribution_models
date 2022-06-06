"""
Element functions
"""


#######################################################
# Modules:
import numpy as np

# Local modules:
from basic_tools import get_kwarg_value
from evolution_models.tools import Approximation_ell


#######################################################
# Function to get element vectors from global vector:
def get_element_vector(global_vector, ell, Np):
    elemental_vector = global_vector[ell * Np: (ell + 1) * Np]
    return elemental_vector


#######################################################
# Function to get element matrices from global matrix:
def get_element_matrix(global_matrix, ell, Np):
    element_matrix = global_matrix[ell * Np: (ell + 1) * Np, ell * Np: (ell + 1) * Np]
    return element_matrix


#######################################################
# Function that returns elemental function approximation f^ell(x, t):
def get_ell_function(coefficient, phi, ell, Np):
    coefficient_ell = get_element_vector(coefficient, ell, Np)  # Obtaining element coefficients
    phi_ell = get_element_vector(phi, ell, Np)  # Obtaining element basis functions

    def func_ell_eval(x):  # Function that evaluates func_ell
        return Approximation_ell(coefficient_ell, phi_ell, Np).eval(x)
    return func_ell_eval  # Returning f^ell(x, t).eval(x) function


#######################################################
# Class which returns vector function Phi(x):
class Phi_vector:

    def __init__(self, phi, N, **kwargs):
        self.phi = phi  # Basis functions
        self.N = N  # Total degrees of freedom
        self.return_2D = get_kwarg_value(kwargs, 'return_2D', False)  # Set to True to return 2-D array (as column vector)

    def get(self, x):  # Vector function Phi(x)
        output = np.zeros(self.N)
        for i in range(self.N):
            output[i] = self.phi[i](x)
        if self.return_2D:
            output = np.reshape(output, (self.N, 1))
        return output


#######################################################
# Class which returns vector function Phi^ell(x):
class Phi_ell_vector:

    def __init__(self, phi, ell, Np, **kwargs):
        self.phi_ell = get_element_vector(phi, ell, Np)  # Element basis functions
        self.Np = Np  # Number of basis functions in each element
        self.return_2D = get_kwarg_value(kwargs, 'return_2D', False)  # Set to True to return 2-D array (as column vector)

    def get(self, x):  # Vector function Phi^ell(x)
        output = np.zeros(self.Np)
        for i in range(self.Np):
            output[i] = self.phi_ell[i](x)
        if self.return_2D:
            output = np.reshape(output, (self.Np, 1))
        return output

"""
Miscellaneous functions
"""


#######################################################
# Modules:
from numpy import zeros, exp, arange

# Local modules:
from basic_tools import get_kwarg_value


#######################################################
# Function to compute simple covariance matrix with Legendre polynomial structure:
def compute_simple_covariance_matrix(N, Np, sigma):
    matrix = zeros([N, N])  # Initialising
    for i in range(N):
        for j in range(Np):
            if i % Np == j:
                matrix[i, i] = sigma[j] ** 2
    return matrix


#######################################################
# Function to compute correlated covariance matrix with Legendre polynomial structure:
def compute_correlated_covariance_matrix(N, Np, Ne, sigma, correlation_strength, **kwargs):
    use_element_multiplier = get_kwarg_value(kwargs, 'use_element_multiplier', True)
    matrix = zeros([N, N])  # Initialising
    var = sigma ** 2  # Variance computation
    element_multiplier = (1 / (Ne + 1)) * arange(Ne + 1, 1, -1)  # Multiplier as element increases
    # Iterating over elements:
    for ell_i in range(Ne):
        for ell_j in range(Ne):
            # Iterating over elements (for multiplier):
            for element in range(Ne):
                # Checking element multiplier:
                if abs(ell_i - ell_j) % Ne == element:
                    # Iterating over polynomial degrees:
                    for degree_i in range(Np):
                        for degree_j in range(Np):
                            # If polynomial degrees are the same:
                            if degree_i == degree_j:
                                i = ell_i * Np + degree_i
                                j = ell_j * Np + degree_j
                                if use_element_multiplier:
                                    matrix[i, j] = element_multiplier[ell_i] * element_multiplier[ell_j] * var[degree_i] * exp(-(1 / correlation_strength) * element)
                                else:
                                    matrix[i, j] = var[degree_i] * exp(-(1 / correlation_strength) * element)
    return matrix

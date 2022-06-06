"""
Legendre polynomials and associated functions
"""


#######################################################
# Modules:
import numpy as np
from math import factorial as ft
import numpy.polynomial.polynomial as poly

# Local modules:
import basic_tools.polynomial_approximators as local_poly


#######################################################
# Legendre polynomial function; P_n(x) = Legendre(x, n):
def Legendre(x, n, **kwargs):
    # Normalisation:
    normalise_constant = 1  # Default is not normalised
    if 'normalise' in kwargs:
        if kwargs['normalise']:
            normalise_constant = np.sqrt(2 / (2 * n + 1))  # Normalisation constant

    # Factorial function:
    def fact(n, k):
        return ft(n) / (ft(k) * ft(n - k))

    # Binomial representation:
    binomial = np.zeros(n + 1)
    for k in range(0, n + 1):
        binomial[k] = fact(n, k) ** 2 * (x - 1) ** (n - k) * (x + 1) ** k

    # Legendre polynomial P_n(x):
    P_n = (1 / 2 ** n) * sum(binomial)

    # Normalising Legendre polynomial if True:
    output = P_n / normalise_constant

    return output


#######################################################
# Function to compute N Legendre-Gauss-Lobatto points:
def compute_LGL_points(N, **kwargs):
    # Number of discretisation points between [-1, 1]; make higher for better root finding accuracy
    if 'x_evals' in kwargs:
        x_evals = kwargs['x_evals']
    else:
        x_evals = 200  # Default

    # Discretisation between [-1, 1]
    x = np.linspace(-1, 1, x_evals)

    # Legendre polynomial P_n(x):
    n = N - 1  # Order of polynomial:

    def P_n(x):
        return Legendre(x, n)

    # Coefficients of derivative of Legendre polynomial; P_n'(x)
    dp = local_poly.polyderivative(P_n, -1, 1, n)

    # Roots of f(x) are the LGL points:
    def f(x):
        return (1 - x ** 2) * poly.polyval(x, dp)

    # Polynomial coefficients of f:
    fy = np.zeros(x_evals)
    for i in range(x_evals):
        fy[i] = f(x[i])
    fp = poly.polyfit(x, fy, n + 1)

    # Computing roots of f(x), hence LGL points:
    LGL_points = poly.polyroots(fp)

    return LGL_points

"""
General polynomial functions
"""


#######################################################
# Modules:
import numpy as np
import numpy.polynomial.polynomial as poly


#######################################################
# Function to compute coefficients of derivative of n-th polynomial
# approximation of 1-D function f(x) between [a, b]:
def polyderivative(f, a, b, n, **kwargs):
    # Number of discretisation points between [a, b]; make higher for better root finding accuracy:
    if 'x_evals' in kwargs:
        x_evals = kwargs['x_evals']
    else:
        x_evals = 200  # Default

    # Discretisation between [a, b]:
    x = np.linspace(a, b, x_evals)

    # Coefficients of polynomial approximation of f(x):
    y = np.zeros(x_evals)
    for i in range(x_evals):
        y[i] = f(x[i])
    p = poly.polyfit(x, y, n)

    # Coefficients of derivative of polynomial approximation:
    dp = poly.polyder(p)

    return dp

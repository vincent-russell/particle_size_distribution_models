"""
Numerical integration functions
"""


#######################################################
# Modules:
import numpy.polynomial.legendre as npLF


#######################################################
# n-point Gauss-Legendre quadrature of the function f from a to b:
def GLnpt(f, a, b, n):
    # Scaling:
    h = (b - a)
    c = (b + a) / 2
    # Compute Gauss nodes and weights:
    x, w = npLF.leggauss(n)
    # Integral approximation:
    int_approx = 0  # Initialising
    for i in range(n):
        int_approx += w[i] * f((h / 2) * x[i] + c)  # Summing terms at each node
    output = (h / 2) * int_approx  # Final scaling
    return output

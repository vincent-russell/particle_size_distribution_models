"""
Miscellaneous functions
"""


#######################################################
# Modules:
import numpy as np
from scipy.special import erf


#######################################################
# Returns value of keyword argument if default or not:
def get_kwarg_value(kwargs, kwarg_name, default_value):
    if kwarg_name in kwargs:
        value = kwargs[kwarg_name]
    else:
        value = default_value
    return value


#######################################################
# Class with Zero.function:H->R such that Zero.function(x) = 0 where H is any space
# and Zero.function_array:H->R^{NxM} such that Zero.function_array(x) = 0_{NxM}
class Zero:
    def __init__(self, **kwargs):
        # Dimensions for function array:
        self.N = get_kwarg_value(kwargs, 'N', 1)
        self.M = get_kwarg_value(kwargs, 'M', 1)

    def function(*x):
        return 0

    def function_array(self, *x):
        return np.zeros([self.N, self.M])


#######################################################
# Removes negative values and sets to zero:
def positive(x):
    if x < 0:
        x = 0  # Set to zero
    return x


#######################################################
# Returns name of numpy array by searching in namespace:
def get_array_name(array, namespace):
    return [name for name in namespace if namespace[name] is array][0]


#######################################################
# Gaussian function evaluated at vector of values x with amplitude amp, mean mu, and variance sigma^2:
def gaussian(x, amp, mu, sigma):
    n = np.size(x)
    mu = mu * np.ones(n)
    output = np.dot(amp, np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)))
    if n == 1:  # If x is 1-D:
        output = output[0]  # Return float rather than numpy array
    return output


#######################################################
# Skewed gaussian function evaluated at vector of values x with amplitude amp, mean mu, variance sigma^2, and skewness alpha:
def skewed_gaussian(x, amp, mu, sigma, skewness):
    n = np.size(x)
    mu = mu * np.ones(n)
    z = (x - mu) / sigma
    gauss_output = np.dot(amp, np.exp(-z ** 2 / 2))
    erf_output = erf(np.dot(skewness / np.sqrt(2), z))
    skew_output = np.dot(1/2, (1 + erf_output))
    output = (2 / sigma) * gauss_output * skew_output
    if n == 1:  # If x is 1-D:
        output = output[0]  # Return float rather than numpy array
    return output


#######################################################
# Sigmoid function evaluated at vector of values x with amplitude amp, mean mu, and spread sigma:
def sigmoid(x, amp, mu, sigma):
    n = np.size(x)
    mu = mu * np.ones(n)
    output = np.dot(amp, 1 / (1 + np.exp(-(1 / sigma) * (x - mu))))
    if n == 1:  # If x is 1-D:
        output = output[0]  # Return float rather than numpy array
    return output


#######################################################
# Computes uniform discretisation N of y(x) = f(x_i) over x_0, x_1, ..., x_{N - 1}:
def uniform_discretisation_from_function(N, f, a, b):
    x = np.linspace(a, b, N)  # Uniform discretisation of x in [a, b]
    y = np.zeros(N)  # Initialising
    # Iterating of x:
    for i in range(N):
        y[i] = f(x[i])  # Computing corresponding y values over uniform discretisation in x
    return x, y


#######################################################
# Return function by interpolation from points (x_p, y_p):
def discretisation_to_function(x_p, y_p):
    def f(x):
        return np.interp(x, x_p, y_p)
    return f


#######################################################
# Interpolates y_t on x from (xp, yp_t) for t = 0, 1, ..., NT - 1
def time_var_interpolate(x, xp, yp, NT):
    N = np.size(x)  # Length of x
    y = np.zeros([N, NT])  # Initialising
    # Iterating over t:
    for t in range(NT):
        # Iterating of x:
        y[:, t] = np.interp(x, xp, yp[:, t])  # Interpolation
    return y


#######################################################
# Returns values of volume to diameter of a sphere:
def volume_to_diameter(v):
    return np.cbrt(np.dot((6 / np.pi), v))


#######################################################
# Returns values of volume to diameter of a sphere:
def diameter_to_volume(d):
    return np.dot((np.pi / 6), (d ** 3))

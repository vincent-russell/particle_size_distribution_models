"""
Time integration functions
"""


#######################################################
# Local modules:
from basic_tools import get_kwarg_value


#######################################################
# Performs one step of Euler's method, where x' = f(x, ...), with time step dt:
def euler(f, x_t, dt, *t, **kwargs):
    f_args = get_kwarg_value(kwargs, 'f_args', x_t)  # Function arguments
    if len(t) == 0:  # Not time dependent
        x_t_plus_1 = x_t + dt * f(f_args)
    else:  # Time dependent
        t = t[0]  # Extracting time from tuple *t
        x_t_plus_1 = x_t + dt * f(f_args, t)
    return x_t_plus_1


#######################################################
# Performs one step of Runge-Kutta order 4, where x' = f(x), with time step dt:
def rk4(f, x_t, dt, *t):
    if len(t) == 0:  # Not time dependent
        k1 = dt * f(x_t)
        k2 = dt * f(x_t + k1 / 2)
        k3 = dt * f(x_t + k2 / 2)
        k4 = dt * f(x_t + k3)
        x_t_plus_1 = x_t + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    else:  # Time dependent
        t = t[0]  # Extracting time from tuple *t
        k1 = dt * f(x_t, t)
        k2 = dt * f(x_t + k1 / 2, t + dt / 2)
        k3 = dt * f(x_t + k2 / 2, t + dt / 2)
        k4 = dt * f(x_t + k3, t + dt)
        x_t_plus_1 = x_t + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_t_plus_1

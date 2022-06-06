"""

Title: Wave simulated by the advection equation using the Discontinuous-Galerkin method
Author: Vincent Russell
Date: August 21, 2021

"""


#######################################################
# Modules:
import numpy as np
import time as tm
from tkinter import mainloop
from tqdm import tqdm

# Local modules:
import basic_tools
import basic_models.advection_models.tools as tools


#######################################################
if __name__ == '__main__':

    #######################################################
    # Parameters:

    # Setup and plotting:
    plot_animations = True  # Set to True to plot animations
    use_crank_nicolson = True  # Set to True to use Crank-Nicolson method

    # Spatial domain:
    xmin = 0  # Minimum
    xmax = 1  # Maximum

    # Time domain:
    dt = 0.001  # Time step
    T = 1  # End time
    NT = int(T / dt)  # Total number of time steps

    # Solution discretisation:
    Ne = 50  # Number of elements
    Np = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
    N = Ne * Np  # Total degrees of freedom

    # Initial condition n_0(x) = n(x, 0):
    N_0 = 1  # Amplitude of initial condition gaussian
    mu_0 = 0.2  # Mean of initial condition gaussian
    sigma_0 = 0.04  # Standard deviation of initial condition gaussian
    def initial_condition(x):
        return basic_tools.gaussian(x, N_0, mu_0, sigma_0)

    # Advection model:
    c_0 = 0  # Constant coefficient
    c_1 = 3.5  # Linear coefficient
    c_2 = -3.5  # Quadratic coefficient
    def advection(x):
        return c_0 + c_1 * x + c_2 * x ** 2


    #######################################################
    # Initialising timer for total computation:
    basic_tools.print_lines()
    initial_time = tm.time()  # Initial time stamp


    #######################################################
    # Constructing evolution model:
    print('Constructing evolution model...')
    # Discretisation:
    x_boundaries, h = tools.get_discretisation(Ne, xmin, xmax)
    # Basis functions:
    phi = tools.get_Legendre_basis(N, Np, x_boundaries)
    dphi = tools.get_Legendre_basis_derivative(N, Np, x_boundaries, phi)
    # Tensors:
    M = tools.compute_M(N, Np, h)
    Q = tools.compute_Q(advection, N, Np, x_boundaries, phi, dphi)
    R = tools.compute_R(advection, Ne, Np, N, x_boundaries, phi, M, Q)
    # Evolution model:
    if use_crank_nicolson:
        F = np.matmul(np.linalg.inv(2 * np.eye(N) - dt * R), 2 * np.eye(N) + dt * R)  # Using Crank-Nicolson method
    else:
        F = np.eye(N) + dt * R  # Using Forward Euler's method


    #######################################################
    # Computing prior:
    alpha = np.zeros([N, NT])  # Initialising alpha = [alpha_0, alpha_1, ..., alpha_NT]
    alpha[:, 0] = tools.compute_coefficients(initial_condition, N, Np, phi, x_boundaries, h)  # Computing coefficients from initial condition function


    #######################################################
    # Computing time evolution of model:
    print('Computing time evolution...')
    t = np.zeros(NT)  # Initialising time array
    for k in tqdm(range(NT - 1)):  # Iterating over time
        alpha[:, k + 1] = np.matmul(F, alpha[:, k])  # Computing next step
        t[k + 1] = (k + 1) * dt  # Time (hours)


    #######################################################
    # Computing plotting discretisation:
    print('Computing plotting discretisation...')
    x_plot, n_plot = tools.get_plotting_discretisation(alpha, xmin, xmax, NT, phi, Np, Ne, x_boundaries, h, 100)


    #######################################################
    # Computing advection plotting discretisation:
    advection_plot = np.zeros([100, NT])
    for k in range(NT):  # Iterating over time
        for i in range(100):  # Iterating over space
            advection_plot[i, k] = advection(x_plot[i])


    #######################################################
    # Printing total computation time:
    computation_time = round(tm.time() - initial_time, 3)  # Initial time stamp
    print('Total computation time:', str(computation_time), 'seconds.')  # Print statement


    #######################################################
    # Plotting size distribution animation and function parameters:

    # General parameters:
    location = 'Home'  # Set to 'Uni', 'Home', or 'Middle' (default)

    # Parameters for wave animation:
    xscale = 'linear'  # x-axis scaling ('linear' or 'log')
    xlimits = [xmin, xmax]  # Plot boundary limits for x-axis
    ylimits = [-0.2, 1.2]  # Plot boundary limits for y-axis
    xlabel = '$x$'  # x-axis label for 1D animation plot
    ylabel = '$n(x, t)$'  # y-axis label for 1D animation plot
    title = 'Advection by DG method'  # Title for 1D animation plot
    line_color = ['blue']  # Colors of lines in plot
    time = t  # Array where time[i] is plotted (and animated)
    timetext = ('Time = ', '')  # Tuple where text to be animated is: timetext[0] + 'time[i]' + timetext[1]
    delay = 0  # Speed of animation (higher = slower)

    # Parameters for advection animation:
    xscale_advection = 'linear'  # x-axis scaling ('linear' or 'log')
    xlimits_advection = [xmin, xmax]  # Plot boundary limits for x-axis
    ylimits_advection = [0, 1]  # Plot boundary limits for y-axis
    xlabel_advection = '$x$'  # x-axis label for 1D animation plot
    ylabel_advection = '$c(x)$'  # y-axis label for 1D animation plot
    title_advection = 'Advection function'  # Title for 1D animation plot
    location_advection = location + '2'  # Location for plot
    line_color_advection = ['blue']  # Colors of lines in plot
    time_advection = t  # Array where time[i] is plotted (and animated)
    timetext_advection = ('Time = ', '')  # Tuple where text to be animated is: timetext[0] + 'time[i]' + timetext[1]
    delay_advection = 0  # Speed of animation (higher = slower)

    # Wave animation:
    basic_tools.plot_1D_animation(x_plot, n_plot, xlimits=xlimits, ylimits=ylimits, xscale=xscale, xlabel=xlabel, ylabel=ylabel, title=title,
                                  location=location, line_color=line_color, time=time, timetext=timetext, doing_mainloop=False, delay=delay)

    # Advection animation:
    basic_tools.plot_1D_animation(x_plot, advection_plot, xlimits=xlimits_advection, ylimits=ylimits_advection, xscale=xscale_advection, xlabel=xlabel_advection, ylabel=ylabel_advection, title=title_advection,
                                  location=location_advection, line_color=line_color_advection, time=time_advection, timetext=timetext_advection, doing_mainloop=False, delay=delay_advection)

    # Mainloop:
    if plot_animations:
        print('Plotting...')
        mainloop()  # Runs tkinter GUI for plots and animations

    # Final prints:
    basic_tools.print_lines()  # Print lines in console
    print()  # Print space in console

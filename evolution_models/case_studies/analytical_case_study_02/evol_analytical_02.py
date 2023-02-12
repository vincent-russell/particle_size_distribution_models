"""

Title: Computes solution approximations to the condensation equation in log-space.
Author: Vincent Russell
Date: June 7, 2022

"""


#######################################################
# Modules:
import numpy as np
import time as tm
from tkinter import mainloop
from tqdm import tqdm

# Local modules:
import basic_tools
from evolution_models.tools import GDE_evolution_model, change_basis_x_to_logDp, change_basis_ln_to_linear, change_basis_volume_to_diameter


#######################################################
if __name__ == '__main__':

    #######################################################
    # Parameters:
    # Setup and plotting:
    plot_animations = True  # Set to True to plot animations
    discretise_with_diameter = False  # Set to True to discretise with diameter

    # Spatial domain:
    Dp_min = 0.1  # Minimum diameter of particles (micro m)
    Dp_max = 1  # Maximum diameter of particles (micro m)
    vmin = basic_tools.diameter_to_volume(Dp_min)  # Minimum volume of particles (micro m^3)
    vmax = basic_tools.diameter_to_volume(Dp_max)  # Maximum volume of particles (micro m^3)
    xmin = np.log(vmin)  # Lower limit in log-size
    xmax = np.log(vmax)  # Upper limit in log-size

    # Time domain:
    dt = 0.1  # Time step (hours)
    T = 96  # End time (hours)
    NT = int(T / dt)  # Total number of time steps

    # Size distribution discretisation:
    Ne = 50  # Number of elements
    Np = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
    N = Ne * Np  # Total degrees of freedom

    # Initial condition n_Dp(Dp, 0):
    N_0 = 180  # Amplitude of initial condition gaussian
    d_mean = 0.2  # Mean of initial condition gaussian
    sigma_g = 1.2  # Standard deviation of initial condition gaussian
    def initial_condition_Dp(Dp):
        amp = N_0 / (np.sqrt(2 * np.pi) * Dp * np.log(sigma_g))
        cst = (np.log(Dp / d_mean) ** 2) / (2 * np.log(sigma_g) ** 2)
        return amp * np.exp(-cst)
    def initial_condition(x):
        v = np.exp(x)
        Dp = basic_tools.volume_to_diameter(v)
        cst = v * (2 / (np.pi * Dp ** 2))
        return cst * initial_condition_Dp(Dp)

    # Set to True for imposing boundary condition n(vmin, t) = 0:
    boundary_zero = True

    # Condensation model I_Dp(Dp, t):
    A = 0.001   # Condensation parameter (micro m^2 hour^{-1})
    def cond(Dp):
        return A / Dp


    #######################################################
    # Initialising timer for total computation:
    initial_time = tm.time()  # Initial time stamp


    #######################################################
    # Constructing evolution model:
    F = GDE_evolution_model(Ne, Np, xmin, xmax, dt, NT, boundary_zero=boundary_zero, scale_type='log', discretise_with_diameter=discretise_with_diameter)  # Initialising evolution model
    F.add_process('condensation', cond)  # Adding condensation to evolution model
    F.compile(time_integrator='rk4')  # Compiling evolution model and adding time integrator


    #######################################################
    # Computing time evolution of model:
    print('Computing time evolution...')
    alpha = np.zeros([N, NT])  # Initialising alpha = [alpha_0, alpha_1, ..., alpha_NT]
    alpha[:, 0] = F.compute_coefficients('alpha', initial_condition)  # Computing alpha coefficients from initial condition function
    t = np.zeros(NT)  # Initialising time array
    for k in tqdm(range(NT - 1)):  # Iterating over time
        alpha[:, k + 1] = F.eval(alpha[:, k], t[k])  # Time evolution computation
        t[k + 1] = (k + 1) * dt  # Time (hours)


    #######################################################
    # Computing plotting discretisation:
    d_plot, v_plot, n_x_plot, _ = F.get_nplot_discretisation(alpha)  # Computing plotting discretisation
    x_plot = np.log(v_plot)  # ln(v)-spaced plotting discretisation
    n_v_plot = change_basis_ln_to_linear(n_x_plot, v_plot)
    n_Dp_plot = change_basis_volume_to_diameter(n_v_plot, d_plot)
    n_logDp_plot = change_basis_x_to_logDp(n_x_plot, v_plot, d_plot)


    #######################################################
    # Computing analytical solution:
    print('Computing analytical solution...')
    x_analytical = np.linspace(xmin, xmax, 200)  # Log-discretisation
    v_analytical = np.exp(x_analytical)  # Volume discretisation
    Dp_analytical = basic_tools.volume_to_diameter(v_analytical)  # Diameter discretisation
    n_Dp_analytical = np.zeros([len(v_analytical), NT])  # Initialising
    n_logDp_analytical = np.zeros([len(v_analytical), NT])  # Initialising
    for k in range(1, NT):  # Iterating over time
        for i in range(len(v_analytical)):  # Iterating over volume
            cst = (Dp_analytical[i] ** 2) - (2 * A * t[k])
            if cst < 0:
                cst = 1e-15
            B = (Dp_analytical[i] * N_0) / (np.sqrt(2 * np.pi) * np.log(sigma_g) * cst)
            C = (np.log(np.sqrt(cst) / d_mean) ** 2) / (2 * np.log(sigma_g) ** 2)
            n_Dp_analytical[i, k] = B * np.exp(-C)
            n_logDp_analytical[i, k] = (1 / np.log10(np.e)) * Dp_analytical[i] * n_Dp_analytical[i, k]


    #######################################################
    # Printing total computation time:
    computation_time = round(tm.time() - initial_time, 3)  # Initial time stamp
    print('Total computation time:', str(computation_time), 'seconds.')  # Print statement


    #######################################################
    # Plotting size distribution animation and function parameters:

    # General parameters:
    location = 'Home'  # Set to 'Uni', 'Home', or 'Middle' (default)

    # Parameters for size distribution animation:
    xscale = 'log'  # x-axis scaling ('linear' or 'log')
    xticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  # Plot x-tick labels
    xlimits = [d_plot[0], d_plot[-1]]  # Plot boundary limits for x-axis
    ylimits = [-500, 6000]  # Plot boundary limits for y-axis
    xlabel = '$D_p$ ($\mu$m)'  # x-axis label for 1D animation plot
    ylabel = '$\dfrac{dN}{dlogD_p}$ (cm$^{-3})$'  # y-axis label for 1D animation plot
    title = 'Size distribution'  # Title for 1D animation plot
    legend = ['Numerical solution approximation', 'Analytical solution']  # Adding legend to plot
    legend_position = 'upper left'  # Position of legend
    line_color = ['blue', 'green']  # Colors of lines in plot
    time = t  # Array where time[i] is plotted (and animated)
    timetext = ('Time = ', ' hours')  # Tuple where text to be animated is: timetext[0] + 'time[i]' + timetext[1]
    delay = 0  # Delay between frames in milliseconds

    # Size distribution animation:
    basic_tools.plot_1D_animation(d_plot, n_logDp_plot, plot_add=(Dp_analytical, n_logDp_analytical), xticks=xticks, xlimits=xlimits, ylimits=ylimits, xscale=xscale, xlabel=xlabel, ylabel=ylabel, title=title,
                                  delay=delay, location=location, time=time, timetext=timetext, line_color=line_color, legend=legend, legend_position=legend_position, doing_mainloop=False)

    # Mainloop and print:
    if plot_animations:
        print('Plotting animations...')
        mainloop()  # Runs tkinter GUI for plots and animations


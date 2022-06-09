"""

Title: Computes solution approximations to the coagulation equation.
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
from evolution_models.tools import GDE_evolution_model


#######################################################
if __name__ == '__main__':

    #######################################################
    # Parameters:
    # Setup and plotting:
    plot_animations = True  # Set to True to plot animations
    plot_images = False  # Set to True to plot images
    load_coagulation = False  # Set to True to load coagulation tensors
    save_coagulation = False  # Set to True to save coagulation tensors

    # Spatial domain:
    vmin = 1e-6  # Minimum volume of particles (micro m^3)
    vmax = 1e-3  # Maximum volume of particles (micro m^3)
    xmin = np.log(vmin)  # Lower limit in log-size
    xmax = np.log(vmax)  # Upper limit in log-size

    # Time domain:
    dt = 0.1  # Time step (hours)
    T = 96  # End time (hours)
    NT = int(T / dt)  # Total number of time steps

    # Size distribution discretisation:
    Ne = 10  # Number of elements
    Np = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
    N = Ne * Np  # Total degrees of freedom

    # Initial condition n_v(v, 0):
    N_0 = 1e4  # Total initial number of particles (particles per cm^3)
    v_0 = 2e-5  # Mean initial volume (micro m^3)
    def initial_condition_v(v):
        return ((N_0 * v) / (v_0 ** 2)) * np.exp(-v / v_0)
    def initial_condition(x):
        v = np.exp(x)
        return v * initial_condition_v(v)

    # Set to True for imposing boundary condition n(vmin, t) = 0:
    boundary_zero = True

    # Coagulation model:
    beta_0 = 8e-6  # Coagulation parameter (cm^3 hour^-1)
    def coag(*_):
        return beta_0


    #######################################################
    # Initialising timer for total computation:
    initial_time = tm.time()  # Initial time stamp


    #######################################################
    # Constructing evolution model:
    F = GDE_evolution_model(Ne, Np, xmin, xmax, dt, NT, boundary_zero=boundary_zero, scale_type='log')  # Initialising evolution model
    F.add_process('coagulation', coag, load_coagulation=load_coagulation, save_coagulation=save_coagulation)  # Adding coagulation to evolution model
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
    _, v_plot, n_x_plot, _ = F.get_nplot_discretisation(alpha)  # Computing plotting discretisation


    #######################################################
    # Computing analytical solution:
    print('Computing analytical solution...')
    x_analytical = np.linspace(xmin, xmax, 200)  # Log-discretisation for analytical solution
    v_analytical = np.exp(x_analytical)  # Discretisation for analytical solution
    n_v_analytical = np.zeros([len(v_analytical), NT])  # Initialising
    n_x_analytical = np.zeros([len(v_analytical), NT])  # Initialising
    for k in range(1, NT):  # Iterating over time
        M_T = (2 * N_0) / (2 + (beta_0 * N_0 * t[k]))
        N_T = 1 - (M_T / N_0)
        for i in range(len(v_analytical)):  # Iterating over volume
            A = ((1 - N_T) ** 2 / np.sqrt(N_T)) * (N_0 / v_0)
            B = v_analytical[i] / v_0
            C = (v_analytical[i] * np.sqrt(N_T)) / v_0
            n_v_analytical[i, k] = A * np.exp(-B) * np.sinh(C)
            n_x_analytical[i, k] = n_v_analytical[i, k] * v_analytical[i]


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
    xticks = [1e-6, 1e-5, 1e-4, 1e-3]  # Plot x-tick labels
    xticklabels = ['$10^{-6}$', '$10^{-5}$', '$10^{-4}$', '$10^{-3}$']  # Plot x-tick labels
    xlimits = [vmin, vmax]  # Plot boundary limits for x-axis
    ylimits = [-3000, 6000]  # Plot boundary limits for y-axis
    xlabel = '$v$ ($\mu$m$^3$)'  # x-axis label for 1D animation plot
    ylabel = '$\dfrac{dN}{d\ln(v)}$ (cm$^{-3})$'  # y-axis label for 1D animation plot
    title = 'Size distribution'  # Title for 1D animation plot
    legend = ['Numerical solution approximation', 'Analytical solution']  # Adding legend to plot
    line_color = ['blue', 'green']  # Colors of lines in plot
    line_style = ['solid', 'solid']  # Style of lines in plot
    time = t  # Array where time[i] is plotted (and animated)
    timetext = ('Time = ', ' hours')  # Tuple where text to be animated is: timetext[0] + 'time[i]' + timetext[1]
    delay = 0  # Delay between frames in milliseconds

    # Size distribution animation:
    basic_tools.plot_1D_animation(v_plot, n_x_plot, plot_add=(v_analytical, n_x_analytical), xticks=xticks, xticklabels=xticklabels, xlimits=xlimits, ylimits=ylimits, xscale=xscale, xlabel=xlabel, ylabel=ylabel, title=title,
                                  delay=delay, location=location, legend=legend, time=time, timetext=timetext, line_color=line_color, line_style=line_style, doing_mainloop=False)

    # Mainloop and print:
    if plot_animations:
        print('Plotting animations...')
        mainloop()  # Runs tkinter GUI for plots and animations


    #######################################################
    # Static plots:
    import matplotlib.pyplot as plt

    plt.plot(v_plot, n_x_plot[:, -1], 'b')
    plt.plot(v_plot, n_x_plot[:, 480], 'b')
    plt.plot(v_plot, n_x_plot[:, 1], 'b')
    plt.plot(v_analytical, n_x_analytical[:, -1], 'g')
    plt.plot(v_analytical, n_x_analytical[:, 480], 'g')
    plt.plot(v_analytical, n_x_analytical[:, 1], 'g')
    plt.xscale(xscale)
    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.grid()


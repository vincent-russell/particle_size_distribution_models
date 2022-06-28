"""

Title: Computes solution approximations to the general dynamic equation of aerosols
Author: Vincent Russell
Date: June 22, 2022

"""


#######################################################
# Modules:
import numpy as np
import time as tm
import matplotlib.pyplot as plt
from tkinter import mainloop
from tqdm import tqdm

# Local modules:
import basic_tools
from evolution_models.tools import Fuchs_Brownian, GDE_evolution_model, GDE_Jacobian, change_basis_x_to_logDp, change_basis_x_to_logDp_sorc


#######################################################
if __name__ == '__main__':

    #######################################################
    # Parameters:

    # Setup and plotting:
    plot_animations = True  # Set to True to plot animations
    plot_nucleation = False  # Set to True to plot nucleation plot
    plot_images = False  # Set to True to plot images
    load_coagulation = True  # Set to True to load coagulation tensors
    save_coagulation = False  # Set to True to save coagulation tensors
    coagulation_suffix = '0004_to_1_micro_metres'  # Suffix of saved coagulation tensors file

    # Spatial domain:
    Dp_min = 0.004  # Minimum diameter of particles (micro m)
    Dp_max = 1  # Maximum diameter of particles (micro m)
    vmin = basic_tools.diameter_to_volume(Dp_min)  # Minimum volume of particles (micro m^3)
    vmax = basic_tools.diameter_to_volume(Dp_max)  # Maximum volume of particles (micro m^3)
    xmin = np.log(vmin)  # Lower limit in log-size
    xmax = np.log(vmax)  # Upper limit in log-size

    # Time domain:
    dt = (1 / 60) * 20  # Time step (hours)
    T = 24  # End time (hours)
    NT = int(T / dt)  # Total number of time steps

    # Size distribution discretisation:
    Ne = 50  # Number of elements
    Np = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
    N = Ne * Np  # Total degrees of freedom

    # Initial condition n_0(x) = n(x, 0):
    N_0 = 1e3  # Amplitude of initial condition gaussian
    x_0 = np.log(basic_tools.diameter_to_volume(0.01))  # Mean of initial condition gaussian
    sigma_0 = 3  # Standard deviation of initial condition gaussian
    skewness = 3  # Skewness factor for initial condition gaussian
    def initial_condition(x):
        return basic_tools.skewed_gaussian(x, N_0, x_0, sigma_0, skewness)

    # Set to True for imposing boundary condition n(xmin, t) = 0:
    boundary_zero = True

    # Condensation model I_Dp(Dp, t):
    I_cst = 0.002  # Condensation parameter constant
    I_linear = 0.05  # Condensation parameter linear
    def cond(Dp):
        return I_cst + I_linear * Dp

    # Deposition model d(Dp, t):
    d_cst = 0.02  # Deposition parameter constant
    d_linear = 0.05  # Deposition parameter linear
    d_inverse_quadratic = 0.00001  # Deposition parameter inverse quadratic
    def depo(Dp):
        return d_cst + d_linear * Dp + d_inverse_quadratic * (1 / Dp ** 2)

    # Source (nucleation event) model:
    N_s = 2e3  # Amplitude of gaussian nucleation event
    t_s = 8  # Mean time of gaussian nucleation event
    sigma_s = 1.5   # Standard deviation time of gaussian nucleation event
    def sorc(t):  # Source (nucleation) at xmin
        return basic_tools.gaussian(t, N_s, t_s, sigma_s)  # Gaussian source (nucleation event) model output

    # Coagulation model:
    def coag(x, y):
        v_x = np.exp(x)  # Volume of particle x (micro m^3)
        v_y = np.exp(y)  # Volume of particle y (micro m^3)
        Dp_x = basic_tools.volume_to_diameter(v_x)  # Diameter of particle x (micro m)
        Dp_y = basic_tools.volume_to_diameter(v_y)  # Diameter of particle y (micro m)
        return Fuchs_Brownian(Dp_x, Dp_y)


    #######################################################
    # Initialising timer for total computation:
    initial_time = tm.time()  # Initial time stamp


    #######################################################
    # Constructing evolution model:
    F = GDE_evolution_model(Ne, Np, xmin, xmax, dt, NT, boundary_zero=boundary_zero, scale_type='log')  # Initialising evolution model
    F.add_process('condensation', cond)  # Adding condensation to evolution model
    F.add_process('deposition', depo)  # Adding deposition to evolution model
    F.add_process('source', sorc)  # Adding source to evolution model
    F.add_process('coagulation', coag, load_coagulation=load_coagulation, save_coagulation=save_coagulation, coagulation_suffix=coagulation_suffix)  # Adding coagulation to evolution model
    F.compile()  # Compiling evolution model


    #######################################################
    # Constructing Jacobian of size distribution evolution model:
    J_F = GDE_Jacobian(F)  # Evolution Jacobian


    #######################################################
    # Computing initial condition:
    alpha = np.zeros([N, NT])  # Initialising alpha = [alpha_0, alpha_1, ..., alpha_NT]
    alpha[:, 0] = F.compute_coefficients('alpha', initial_condition)  # Computing alpha coefficients from initial condition function


    #######################################################
    # Function to compute evolution operator using Crank-Nicolson method:
    def compute_evolution_operator(alpha_star, t_star):
        J_star = J_F.eval_d_alpha(alpha_star, t_star)  # Computing J_star
        F_star = F.eval(alpha_star, t_star) - np.matmul(J_star, alpha_star)  # Computing F_star
        matrix_multiplier = np.linalg.inv(np.eye(N) - (dt / 2) * J_star)  # Computing matrix multiplier for evolution operator and additive vector
        F_evolution = np.matmul(matrix_multiplier, (np.eye(N) + (dt / 2) * J_star))  # Computing evolution operator
        b_evolution = np.matmul(matrix_multiplier, (dt * F_star))  # Computing evolution additive vector
        return F_evolution, b_evolution


    #######################################################
    # Computing time evolution of model using Crank-Nicolson method:
    print('Computing time evolution...')
    t = np.zeros(NT)  # Initialising time array
    F_evolution, b_evolution = compute_evolution_operator(alpha[:, 0], t[0])  # Computing evolution operator from initial condition
    for k in tqdm(range(NT - 1)):  # Iterating over time
        F_evolution, b_evolution = compute_evolution_operator(alpha[:, k], t[k])  # Computing evolution operator F and vector b
        alpha[:, k + 1] = np.matmul(F_evolution, alpha[:, k]) + b_evolution  # Time evolution computation
        t[k + 1] = (k + 1) * dt  # Time (hours)


    #######################################################
    # Computing plotting discretisation:
    d_plot, v_plot, n_x_plot, _ = F.get_nplot_discretisation(alpha)  # Computing plotting discretisation
    x_plot = np.log(v_plot)  # ln(v)-spaced plotting discretisation
    n_logDp_plot = change_basis_x_to_logDp(n_x_plot, v_plot, d_plot)  # Computing log_10(D_p)-based size distribution


    #######################################################
    # Computing parameters plotting discretisation:
    Nplot = len(d_plot)  # Length of size discretisation
    cond_Dp_plot = np.zeros([Nplot, NT])  # Initialising ln(volume)-based condensation rate
    depo_plot = np.zeros([Nplot, NT])  # Initialising deposition rate
    sorc_x_plot = np.zeros(NT)  # Initialising ln(volume)-based source (nucleation) rate
    for k in range(NT):
        sorc_x_plot[k] = sorc(t[k])  # Computing ln(volume)-based nucleation rate
        for i in range(Nplot):
            cond_Dp_plot[i, k] = cond(d_plot[i])  # Computing ln(volume)-based condensation rate
            depo_plot[i, k] = depo(d_plot[i])  # Computing deposition rate
    sorc_logDp_plot = change_basis_x_to_logDp_sorc(sorc_x_plot, vmin, Dp_min)  # Computing log_10(D_p)-based nucleation rate


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
    xticks = [0.01, 0.1, 1]  # Plot x-tick labels
    xlimits = [d_plot[0], d_plot[-1]]  # Plot boundary limits for x-axis
    ylimits = [0, 10000]  # Plot boundary limits for y-axis
    xlabel = '$D_p$ ($\mu$m)'  # x-axis label for 1D animation plot
    ylabel = '$\dfrac{dN}{dlogD_p}$ (cm$^{-3})$'  # y-axis label for 1D animation plot
    title = 'Size distribution'  # Title for 1D animation plot
    line_color = ['blue']  # Colors of lines in plot
    time = t  # Array where time[i] is plotted (and animated)
    timetext = ('Time = ', ' hours')  # Tuple where text to be animated is: timetext[0] + 'time[i]' + timetext[1]
    delay = 60  # Delay between frames in milliseconds

    # Parameters for condensation plot:
    yscale_cond = 'linear'  # y-axis scaling ('linear' or 'log')
    ylimits_cond = [0, 0.06]  # Plot boundary limits for y-axis
    xlabel_cond = '$D_p$ ($\mu$m)'  # x-axis label for plot
    ylabel_cond = '$I(D_p)$ ($\mu$m hour$^{-1}$)'  # y-axis label for plot
    title_cond = 'Condensation rate'  # Title for plot
    location_cond = location + '2'  # Location for plot
    line_color_cond = ['blue']  # Colors of lines in plot

    # Parameters for deposition plot:
    yscale_depo = 'linear'  # y-axis scaling ('linear' or 'log')
    ylimits_depo = [0, 0.6]  # Plot boundary limits for y-axis
    xlabel_depo = '$D_p$ ($\mu$m)'  # x-axis label for plot
    ylabel_depo = '$d(D_p)$ (hour$^{-1}$)'  # y-axis label for plot
    title_depo = 'Deposition rate'  # Title for plot
    location_depo = location + '3'  # Location for plot
    line_color_depo = ['blue']  # Colors of lines in plot

    # Size distribution animation:
    basic_tools.plot_1D_animation(d_plot, n_logDp_plot, xticks=xticks, xlimits=xlimits, ylimits=ylimits, xscale=xscale, xlabel=xlabel, ylabel=ylabel, title=title,
                                  delay=delay, location=location, time=time, timetext=timetext, line_color=line_color, doing_mainloop=False)

    # Condensation rate animation:
    basic_tools.plot_1D_animation(d_plot, cond_Dp_plot, xticks=xticks, xlimits=xlimits, ylimits=ylimits_cond, xscale=xscale, yscale=yscale_cond, xlabel=xlabel_cond, ylabel=ylabel_cond, title=title_cond,
                                  location=location_cond, time=time, timetext=timetext, line_color=line_color_cond, doing_mainloop=False)

    # Deposition rate animation:
    basic_tools.plot_1D_animation(d_plot, depo_plot, xticks=xticks, xlimits=xlimits, ylimits=ylimits_depo, xscale=xscale, yscale=yscale_depo, xlabel=xlabel_depo, ylabel=ylabel_depo, title=title_depo,
                                  location=location_depo, time=time, timetext=timetext, line_color=line_color_depo, doing_mainloop=False)

    # Mainloop and print:
    if plot_animations:
        print('Plotting animations...')
        mainloop()  # Runs tkinter GUI for plots and animations


    #######################################################
    # Plotting nucleation rate:
    if plot_nucleation:
        print('Plotting nucleation...')
        figJ, axJ = plt.subplots(figsize=(8.00, 5.00), dpi=100)
        plt.plot(time, sorc_logDp_plot, color='blue')
        axJ.set_xlim([0, T])
        axJ.set_ylim([0, 16000])
        axJ.set_xlabel('$t$ (hour)', fontsize=12)
        axJ.set_ylabel('$J(t)$ \n (cm$^{-3}$ hour$^{-1}$)', fontsize=12, rotation=0)
        axJ.yaxis.set_label_coords(-0.015, 1.02)
        axJ.set_title('Nucleation rate', fontsize=12)
        axJ.grid()


    #######################################################
    # Images:

    # Parameters for size distribution images:
    yscale_image = 'log'  # Change scale of y-axis (linear or log)
    yticks_image = [0.01, 0.1, 1]  # Plot y-tick labels
    xlabel_image = 'Time (hours)'  # x-axis label for image
    ylabel_image = '$D_p$ ($\mu$m) \n'  # y-axis label for image
    ylabelcoords = (-0.06, 0.96)  # y-axis label coordinates
    title_image = 'Size distribution'  # Title for image
    image_min = 10  # Minimum of image colour
    image_max = 10000  # Maximum of image colour
    cmap = 'jet'  # Colour map of image
    cbarlabel = '$\dfrac{dN}{dlogD_p}$ (cm$^{-3})$'  # Label of colour bar
    cbarticks = [10, 100, 1000, 10000]  # Ticks of colorbar

    # Plotting images:
    if plot_images:
        print('Plotting images...')
        basic_tools.image_plot(time, d_plot, n_logDp_plot, xlabel=xlabel_image, ylabel=ylabel_image, title=title_image,
                               yscale=yscale_image, ylabelcoords=ylabelcoords, yticks=yticks_image, image_min=image_min, image_max=image_max, cmap=cmap, cbarlabel=cbarlabel, cbarticks=cbarticks)


    # Final print statements
    basic_tools.print_lines()  # Print lines in console
    print()  # Print space in console

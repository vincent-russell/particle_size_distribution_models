"""

Title: Simualtes observations using the general dynamic equation of aerosols
Author: Vincent Russell
Date: June 22, 2022

"""


#######################################################
# Modules:
import numpy as np
import time as tm
from tkinter import mainloop
from tqdm import tqdm

# Local modules:
import basic_tools
from evolution_models.tools import GDE_evolution_model, GDE_Jacobian, change_basis_x_to_logDp


#######################################################
# Importing parameter file:
from observation_models.simulators.observations_07.obs_07_parameters import *


#######################################################
if __name__ == '__main__':

    #######################################################
    # Initialising timer for total computation:
    initial_time = tm.time()  # Initial time stamp


    #######################################################
    # Constructing evolution model:
    F = GDE_evolution_model(Ne, Np, xmin, xmax, dt, NT, boundary_zero=boundary_zero, scale_type='log')  # Initialising evolution model
    F.add_process('condensation', cond)  # Adding condensation to evolution model
    F.add_process('deposition', depo)  # Adding deposition to evolution model
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
    # Computing true plotting discretisation:
    d_true, v_true, n_x_true, _ = F.get_nplot_discretisation(alpha)  # Computing plotting discretisation
    x_true = np.log(v_true)  # ln(v)-spaced plotting discretisation
    n_logDp_true = change_basis_x_to_logDp(n_x_true, v_true, d_true)  # Computing log_10(D_p)-based size distribution


    #######################################################
    # Computing observation discretisation:
    d_obs = np.exp(logDp_obs)  # Diameters that observations are made
    v_obs = diameter_to_volume(d_obs)  # Volumes that observations are made
    x_obs = np.log(v_obs)  # Log(volume) that observations are made
    _, _, n_x_obs, _ = F.get_nplot_discretisation(alpha, x_plot=x_obs)  # Computing plotting discretisation
    n_logDp_obs = change_basis_x_to_logDp(n_x_obs, v_obs, d_obs)  # Computing log_10(D_p)-based size distribution
    Y = (1 / sample_volume) * basic_tools.get_poisson(sample_volume * n_logDp_obs)  # Drawing observations from Poisson distribution


    #######################################################
    # Computing parameters plotting discretisation:
    Nplot = len(d_true)  # Length of size discretisation
    cond_Dp_plot = np.zeros([Nplot, NT])  # Initialising ln(volume)-based condensation rate
    depo_plot = np.zeros([Nplot, NT])  # Initialising deposition rate
    for k in range(NT):
        for i in range(Nplot):
            cond_Dp_plot[i, k] = cond(d_true[i])  # Computing ln(volume)-based condensation rate
            depo_plot[i, k] = depo(d_true[i])  # Computing deposition rate


    #######################################################
    # Printing total computation time:
    computation_time = round(tm.time() - initial_time, 3)  # Initial time stamp
    print('Total computation time:', str(computation_time), 'seconds.')  # Print statement


    #######################################################
    # Saving observations and true distribution:
    pathname = 'C:/Users/Vincent/OneDrive - The University of Auckland/Python/particle_size_distribution_models/observation_models/data/simulated/' + data_filename  # Adding path to filename
    np.savez(pathname, d_true=d_true, n_true=n_x_true, d_obs=d_obs, Y=Y)  # Saving observation data in .npz file
    print('Saved simulated observations data')


    #######################################################
    # Plotting size distribution animation and function parameters:

    # General parameters:
    location = 'Home'  # Set to 'Uni', 'Home', or 'Middle' (default)

    # Parameters for size distribution animation:
    xscale = 'log'  # x-axis scaling ('linear' or 'log')
    xticks = [0.1, 1, 10]  # Plot x-tick labels
    xlimits = [d_true[0], d_true[-1]]  # Plot boundary limits for x-axis
    ylimits = [0, 10000]  # Plot boundary limits for y-axis
    xlabel = '$D_p$ ($\mu$m)'  # x-axis label for 1D animation plot
    ylabel = '$\dfrac{dN}{dlogD_p}$ (cm$^{-3})$'  # y-axis label for 1D animation plot
    title = 'Size distribution'  # Title for 1D animation plot
    legend = ['True size distribution', 'Simulated observations']  # Adding legend to plot
    line_color = ['blue', 'red']  # Colors of lines in plot
    time = t  # Array where time[i] is plotted (and animated)
    timetext = ('Time = ', ' hours')  # Tuple where text to be animated is: timetext[0] + 'time[i]' + timetext[1]
    delay = 60  # Delay between frames in milliseconds

    # Parameters for condensation plot:
    yscale_cond = 'linear'  # y-axis scaling ('linear' or 'log')
    ylimits_cond = [1e-3, 1e1]  # Plot boundary limits for y-axis
    xlabel_cond = '$D_p$ ($\mu$m)'  # x-axis label for plot
    ylabel_cond = '$I(D_p)$ ($\mu$m hour$^{-1}$)'  # y-axis label for plot
    title_cond = 'Condensation rate'  # Title for plot
    location_cond = location + '2'  # Location for plot
    line_color_cond = ['blue']  # Colors of lines in plot

    # Parameters for deposition plot:
    yscale_depo = 'linear'  # y-axis scaling ('linear' or 'log')
    ylimits_depo = [1e-2, 1e1]  # Plot boundary limits for y-axis
    xlabel_depo = '$D_p$ ($\mu$m)'  # x-axis label for plot
    ylabel_depo = '$d(D_p)$ (hour$^{-1}$)'  # y-axis label for plot
    title_depo = 'Deposition rate'  # Title for plot
    location_depo = location + '3'  # Location for plot
    line_color_depo = ['blue']  # Colors of lines in plot

    # Size distribution animation:
    basic_tools.plot_1D_animation(d_true, n_logDp_true, plot_add=(d_obs, Y), xticks=xticks, xlimits=xlimits, ylimits=ylimits, xscale=xscale, xlabel=xlabel, ylabel=ylabel, title=title,
                                  delay=delay, location=location, legend=legend, time=time, timetext=timetext, line_color=line_color, doing_mainloop=False)

    # Condensation rate animation:
    basic_tools.plot_1D_animation(d_true, cond_Dp_plot, xticks=xticks, xlimits=xlimits, ylimits=ylimits_cond, xscale=xscale, yscale=yscale_cond, xlabel=xlabel_cond, ylabel=ylabel_cond, title=title_cond,
                                  location=location_cond, time=time, timetext=timetext, line_color=line_color_cond, doing_mainloop=False)

    # Deposition rate animation:
    basic_tools.plot_1D_animation(d_true, depo_plot, xticks=xticks, xlimits=xlimits, ylimits=ylimits_depo, xscale=xscale, yscale=yscale_depo, xlabel=xlabel_depo, ylabel=ylabel_depo, title=title_depo,
                                  location=location_depo, time=time, timetext=timetext, line_color=line_color_depo, doing_mainloop=False)

    # Mainloop and print:
    if plot_animations:
        print('Plotting animations...')
        mainloop()  # Runs tkinter GUI for plots and animations


    #######################################################
    # Images:

    # Parameters for size distribution images:
    yscale_image = 'log'  # Change scale of y-axis (linear or log)
    yticks_image = [0.1, 1, 10]  # Plot y-tick labels
    xlabel_image = 'Time (hours)'  # x-axis label for image
    ylabel_image = '$D_p$ ($\mu$m) \n'  # y-axis label for image
    ylabelcoords = (-0.06, 0.96)  # y-axis label coordinates
    title_image = 'Size distribution'  # Title for image
    title_image_observations = 'Simulated observations'  # Title for image
    image_min = 10  # Minimum of image colour
    image_max = 10000  # Maximum of image colour
    cmap = 'jet'  # Colour map of image
    cbarlabel = '$\dfrac{dN}{dlogD_p}$ (cm$^{-3})$'  # Label of colour bar
    cbarticks = [10, 100, 1000, 10000]  # Ticks of colorbar

    # Plotting images:
    if plot_images:
        print('Plotting images...')
        basic_tools.image_plot(time, d_true, n_logDp_true, xlabel=xlabel_image, ylabel=ylabel_image, title=title_image,
                               yscale=yscale_image, ylabelcoords=ylabelcoords, yticks=yticks_image, image_min=image_min, image_max=image_max, cmap=cmap, cbarlabel=cbarlabel, cbarticks=cbarticks)
        basic_tools.image_plot(time, d_obs, Y, xlabel=xlabel_image, ylabel=ylabel_image, title=title_image_observations,
                               yscale=yscale_image, ylabelcoords=ylabelcoords, yticks=yticks_image, image_min=image_min, image_max=image_max, cmap=cmap, cbarlabel=cbarlabel, cbarticks=cbarticks)


    # Final print statements
    basic_tools.print_lines()  # Print lines in console
    print()  # Print space in console

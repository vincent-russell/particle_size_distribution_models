"""

Title: Computes solution approximations to the general dynamic equation of aerosols in log-size (comparing to CSTAR)
Author: Vincent Russell
Date: Auguest 04, 2021

"""


#######################################################
# Modules:
import numpy as np
import time as tm
import matplotlib.pyplot as plt
from tkinter import mainloop

# Local modules:
import basic_tools
from observation_models.CSTAR.files import get_CSTAR_data
from evolution_models.tools import GDE_evolution_model, change_basis_x_to_logDp, change_basis_x_to_logDp_sorc


#######################################################
# Importing parameter file:
from observation_models.simulators.observations_08.obs_08_parameters import *


#######################################################
if __name__ == '__main__':

    #######################################################
    # Initialising timer for total computation:
    initial_time = tm.time()  # Initial time stamp


    #######################################################
    # Importing CSTAR observations:
    d_CSTAR, Y_CSTAR, time_CSTAR = get_CSTAR_data()


    #######################################################
    # Constructing evolution model:
    F = GDE_evolution_model(Ne, Np, xmin, xmax, dt, NT, boundary_zero=boundary_zero, scale_type='log')  # Initialising evolution model
    F.add_process('condensation', cond)  # Adding condensation to evolution model
    F.add_process('deposition', depo)  # Adding deposition to evolution model
    F.add_process('source', sorc)  # Adding source to evolution model
    F.add_process('coagulation', coag, load_coagulation=load_coagulation, coagulation_suffix=coagulation_suffix)  # Adding coagulation to evolution model
    F.compile(time_integrator='rk4')  # Compiling evolution model and adding time integrator


    #######################################################
    # Computing time evolution of model:
    alpha = np.zeros([N, NT])  # Initialising alpha = [alpha_0, alpha_1, ..., alpha_NT]
    alpha[:, 0] = F.compute_coefficients('alpha', initial_condition)  # Computing alpha coefficients from initial condition function
    t = np.zeros(NT)  # Initialising time array
    for k in range(NT - 1):  # Iterating over time
        alpha[:, k + 1] = F.eval(alpha[:, k], t[k])  # Time evolution computation
        t[k + 1] = (k + 1) * dt  # Time (hours)


    #######################################################
    # Computing true plotting discretisation:
    d_true, v_true, n_x_true, _ = F.get_nplot_discretisation(alpha)  # Computing plotting discretisation
    x_true = np.log(v_true)  # ln(v)-spaced plotting discretisation
    n_logDp_true = change_basis_x_to_logDp(n_x_true, v_true, d_true)  # Computing log_10(D_p)-based size distribution


    #######################################################
    # Computing observation discretisation:
    d_obs, v_obs, n_x_obs, _ = F.get_nplot_discretisation(alpha, Nplot=M)  # Computing plotting discretisation over Nplot = M
    n_logDp_obs = change_basis_x_to_logDp(n_x_obs, v_obs, d_obs)  # Computing log_10(D_p)-based size distribution
    Y = (1 / sample_volume) * basic_tools.get_poisson(sample_volume * n_logDp_obs)  # Drawing observations from Poisson distribution


    #######################################################
    # Computing parameters plotting discretisation:
    Nplot = len(d_true)  # Length of size discretisation
    cond_Dp_plot = np.zeros([Nplot, NT])  # Initialising ln(volume)-based condensation rate
    depo_plot = np.zeros([Nplot, NT])  # Initialising deposition rate
    sorc_x_plot = np.zeros(NT)  # Initialising ln(volume)-based source (nucleation) rate
    for k in range(NT):
        sorc_x_plot[k] = sorc(t[k])  # Computing ln(volume)-based nucleation rate
        for i in range(Nplot):
            cond_Dp_plot[i, k] = cond(d_true[i])  # Computing ln(volume)-based condensation rate
            depo_plot[i, k] = depo(d_true[i])  # Computing deposition rate
    sorc_logDp_plot = change_basis_x_to_logDp_sorc(sorc_x_plot, vmin, Dp_min)  # Computing log_10(D_p)-based nucleation rate


    #######################################################
    # Printing total computation time:
    computation_time = round(tm.time() - initial_time, 3)  # Initial time stamp
    print('Total computation time:', str(computation_time), 'seconds.')  # Print statement


    #######################################################
    # Saving observations and true distribution:
    pathname = 'data/' + data_filename  # Adding path to filename
    np.savez(pathname, d_true=d_true, n_true=n_x_true, d_obs=d_obs, Y=Y)  # Saving observation data in .npz file
    if save_parameter_file:
        basic_tools.copy_code('observation_simulator_log_case_02_parameters', pathname)  # Saving parameter.py file as .txt file
    print('Saved simulated observations data')


    #######################################################
    # Plotting size distribution animation and function parameters:

    # General parameters:
    location = 'Home'  # Set to 'Uni', 'Home', or 'Middle' (default)

    # Parameters for size distribution animation:
    xscale = 'log'  # x-axis scaling ('linear' or 'log')
    xticks = [Dp_min, 0.1, Dp_max]  # Plot x-tick labels
    xlimits = [Dp_min, Dp_max]  # Plot boundary limits for x-axis
    ylimits = [0, 300]  # Plot boundary limits for y-axis
    xlabel = '$D_p$ ($\mu$m)'  # x-axis label for 1D animation plot
    ylabel = '$\dfrac{dN}{dlogD_p}$ (cm$^{-3})$'  # y-axis label for 1D animation plot
    title = 'Size distribution'  # Title for 1D animation plot
    legend = ['True size distribution', 'Simulated observations', 'CSTAR observations']  # Adding legend to plot
    line_color = ['blue', 'red', 'purple']  # Colors of lines in plot
    time = t  # Array where time[i] is plotted (and animated)
    timetext = ('Time = ', ' hours')  # Tuple where text to be animated is: timetext[0] + 'time[i]' + timetext[1]
    delay = 15  # Delay between frames in milliseconds

    # Parameters for condensation plot:
    ylimits_cond = [0, 0.01]  # Plot boundary limits for y-axis
    xlabel_cond = '$D_p$ ($\mu$m)'  # x-axis label for plot
    ylabel_cond = '$I(D_p)$ ($\mu$m hour$^{-1}$)'  # y-axis label for plot
    title_cond = 'Condensation rate'  # Title for plot
    location_cond = location + '2'  # Location for plot
    line_color_cond = ['blue']  # Colors of lines in plot

    # Parameters for deposition plot:
    yscale_depo = 'linear'  # y-axis scaling ('linear' or 'log')
    ylimits_depo = [0, 1]  # Plot boundary limits for y-axis
    xlabel_depo = '$D_p$ ($\mu$m)'  # x-axis label for plot
    ylabel_depo = '$d(D_p)$ (hour$^{-1}$)'  # y-axis label for plot
    title_depo = 'Deposition rate'  # Title for plot
    location_depo = location + '3'  # Location for plot
    line_color_depo = ['blue']  # Colors of lines in plot

    # Size distribution animation:
    basic_tools.plot_1D_animation(d_true, n_logDp_true, Y, plot_add=(d_CSTAR, Y_CSTAR), xticks=xticks, xlimits=xlimits, ylimits=ylimits, xscale=xscale, xlabel=xlabel, ylabel=ylabel, title=title,
                                  delay=delay, location=location, legend=legend, time=time, timetext=timetext, line_color=line_color, doing_mainloop=False)

    # Condensation rate animation:
    basic_tools.plot_1D_animation(d_true, cond_Dp_plot, xticks=xticks, xlimits=xlimits, ylimits=ylimits_cond, xscale=xscale, xlabel=xlabel_cond, ylabel=ylabel_cond, title=title_cond,
                                  location=location_cond, time=time, timetext=timetext, line_color=line_color_cond, doing_mainloop=False)

    # Deposition rate animation:
    basic_tools.plot_1D_animation(d_true, depo_plot, xticks=xticks, xlimits=xlimits, ylimits=ylimits_depo, xscale=xscale, yscale=yscale_depo, xlabel=xlabel_depo, ylabel=ylabel_depo, title=title_depo,
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
        axJ.set_ylim([0, 14000])
        axJ.set_xlabel('$t$ (hour)', fontsize=12)
        axJ.set_ylabel('$J(t)$ \n (cm$^{-3}$ hour$^{-1}$)', fontsize=12, rotation=0)
        axJ.yaxis.set_label_coords(-0.015, 1.02)
        axJ.set_title('Nucleation rate', fontsize=12)
        axJ.grid()


    #######################################################
    # Images:

    # Parameters for size distribution images:
    yscale_image = 'log'  # Change scale of y-axis (linear or log)
    yticks_image = [Dp_min, 0.1, Dp_max]  # Plot y-tick labels
    xlabel_image = 'Time (hours)'  # x-axis label for image
    ylabel_image = '$D_p$ ($\mu$m) \n'  # y-axis label for image
    ylabelcoords = (-0.06, 0.96)  # y-axis label coordinates
    title_image = 'Size distribution'  # Title for image
    title_image_observations = 'Simulated observations'  # Title for image
    title_image_CSTAR = 'CSTAR observations'  # Title for image
    image_min = 1  # Minimum of image colour
    image_max = 300  # Maximum of image colour
    cmap = 'jet'  # Colour map of image
    cbarlabel = '$\dfrac{dN}{dlogD_p}$ (cm$^{-3})$'  # Label of colour bar
    cbarticks = [1, 10, 100]  # Ticks of colorbar

    # Plotting images:
    if plot_images:
        print('Plotting images...')
        basic_tools.image_plot(time, d_true, n_logDp_true, xlabel=xlabel_image, ylabel=ylabel_image, title=title_image,
                               yscale=yscale_image, ylabelcoords=ylabelcoords, yticks=yticks_image, image_min=image_min, image_max=image_max, cmap=cmap, cbarlabel=cbarlabel, cbarticks=cbarticks)
        basic_tools.image_plot(time, d_obs, Y, xlabel=xlabel_image, ylabel=ylabel_image, title=title_image_observations,
                               yscale=yscale_image, ylabelcoords=ylabelcoords, yticks=yticks_image, image_min=image_min, image_max=image_max, cmap=cmap, cbarlabel=cbarlabel, cbarticks=cbarticks)
        basic_tools.image_plot(time_CSTAR, d_CSTAR, Y_CSTAR, xlabel=xlabel_image, ylabel=ylabel_image, title=title_image_CSTAR,
                               yscale=yscale_image, ylabelcoords=ylabelcoords, yticks=yticks_image, image_min=image_min, image_max=image_max, cmap=cmap, cbarlabel=cbarlabel, cbarticks=cbarticks)

    # Final print statements
    basic_tools.print_lines()  # Print lines in console
    print()  # Print space in console

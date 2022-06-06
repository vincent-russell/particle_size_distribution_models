"""

Title: Computes solution approximations to the general dynamic equation of aerosols.
Author: Vincent Russell
Date: May 18, 2020

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
from evolution_models.tools import GDE_evolution_model, change_basis_volume_to_diameter, change_basis_volume_to_diameter_sorc


#######################################################
# Importing parameter file:
from observation_models.simulators.observations_01.obs_01_parameters import *


#######################################################
if __name__ == '__main__':

    #######################################################
    # Initialising timer for total computation:
    initial_time = tm.time()  # Initial time stamp


    #######################################################
    # Constructing evolution model:
    F = GDE_evolution_model(Ne, Np, vmin, vmax, dt, NT, boundary_zero=boundary_zero)  # Initialising evolution model
    F.add_process('condensation', cond)  # Adding condensation to evolution model
    F.add_process('deposition', depo)  # Adding deposition to evolution model
    F.add_process('source', sorc)  # Adding source to evolution model
    F.add_process('coagulation', coag, load_coagulation=load_coagulation)  # Adding coagulation to evolution model
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
    # Computing true plotting discretisation:
    d_true, v_true, n_v_true, _ = F.get_nplot_discretisation(alpha)  # Computing plotting discretisation
    n_Dp_true = change_basis_volume_to_diameter(n_v_true, d_true)  # Computing diameter-based size distribution


    #######################################################
    # Computing observation discretisation:
    d_obs, v_obs, n_v_obs, _ = F.get_nplot_discretisation(alpha, Nplot=M)  # Computing plotting discretisation over Nplot = M
    n_Dp_obs = change_basis_volume_to_diameter(n_v_obs, d_obs)  # Computing diameter-based size distribution
    Y = (1 / sample_volume) * basic_tools.get_poisson(sample_volume * n_Dp_obs)  # Drawing observations from Poisson distribution


    #######################################################
    # Computing parameters plotting discretisation:
    Nplot = len(d_true)  # Length of size discretisation
    cond_Dp_plot = np.zeros([Nplot, NT])  # Initialising volume-based condensation rate
    depo_plot = np.zeros([Nplot, NT])  # Initialising deposition rate
    sorc_v_plot = np.zeros(NT)  # Initialising volume-based source (nucleation) rate
    for k in range(NT):
        sorc_v_plot[k] = sorc(t[k])  # Computing volume-based nucleation rate
        for i in range(Nplot):
            cond_Dp_plot[i, k] = cond(d_true[i])  # Computing volume-based condensation rate
            depo_plot[i, k] = depo(d_true[i])  # Computing deposition rate
    sorc_Dp_plot = change_basis_volume_to_diameter_sorc(sorc_v_plot, Dp_min)  # Computing diameter-based nucleation rate


    #######################################################
    # Printing total computation time:
    computation_time = round(tm.time() - initial_time, 3)  # Initial time stamp
    print('Total computation time:', str(computation_time), 'seconds.')  # Print statement


    #######################################################
    # Saving observations and true distribution:
    pathname = 'data/' + data_filename  # Adding path to filename
    np.savez(pathname, d_true=d_true, n_true=n_v_true, d_obs=d_obs, Y=Y)  # Saving observation data in .npz file
    if save_parameter_file:
        basic_tools.copy_code('observation_simulator_parameters', pathname)  # Saving parameter.py file as .txt file
    print('Saved simulated observations data')


    #######################################################
    # Plotting size distribution animation and function parameters:

    # General parameters:
    location = 'Home'  # Set to 'Uni', 'Home', or 'Middle' (default)

    # Parameters for size distribution animation:
    xscale = 'linear'  # x-axis scaling ('linear' or 'log')
    xlimits = [d_true[0], d_true[-1]]  # Plot boundary limits for x-axis
    ylimits = [0, 12000]  # Plot boundary limits for y-axis
    xlabel = '$D_p$ ($\mu$m)'  # x-axis label for 1D animation plot
    ylabel = '$\dfrac{dN}{dD_p}$ $(\mu$m$^{-1}$cm$^{-3})$'  # y-axis label for 1D animation plot
    title = 'Size distribution'  # Title for 1D animation plot
    legend = ['True size distribution', 'Simulated observations']  # Adding legend to plot
    line_color = ['blue', 'red']  # Colors of lines in plot
    time = t  # Array where time[i] is plotted (and animated)
    timetext = ('Time = ', ' hours')  # Tuple where text to be animated is: timetext[0] + 'time[i]' + timetext[1]

    # Parameters for condensation plot:
    ylimits_cond = [0, 2]  # Plot boundary limits for y-axis
    xlabel_cond = '$D_p$ ($\mu$m)'  # x-axis label for plot
    ylabel_cond = '$I(D_p)$ ($\mu$m hour$^{-1}$)'  # y-axis label for plot
    title_cond = 'Condensation rate'  # Title for plot
    location_cond = location + '2'  # Location for plot
    line_color_cond = ['blue']  # Colors of lines in plot

    # Parameters for deposition plot:
    ylimits_depo = [0, 1]  # Plot boundary limits for y-axis
    xlabel_depo = '$D_p$ ($\mu$m)'  # x-axis label for plot
    ylabel_depo = '$d(D_p)$ (hour$^{-1}$)'  # y-axis label for plot
    title_depo = 'Deposition rate'  # Title for plot
    location_depo = location + '3'  # Location for plot
    line_color_depo = ['blue']  # Colors of lines in plot

    # Size distribution animation:
    basic_tools.plot_1D_animation(d_true, n_Dp_true, plot_add=(d_obs, Y), xlimits=xlimits, ylimits=ylimits, xscale=xscale, xlabel=xlabel, ylabel=ylabel, title=title,
                                  location=location, legend=legend, time=time, timetext=timetext, line_color=line_color, doing_mainloop=False)

    # Condensation rate animation:
    basic_tools.plot_1D_animation(d_true, cond_Dp_plot, xlimits=xlimits, ylimits=ylimits_cond, xscale=xscale, xlabel=xlabel_cond, ylabel=ylabel_cond, title=title_cond,
                                  location=location_cond, time=time, timetext=timetext, line_color=line_color_cond, doing_mainloop=False)

    # Deposition rate animation:
    basic_tools.plot_1D_animation(d_true, depo_plot, xlimits=xlimits, ylimits=ylimits_depo, xscale=xscale, xlabel=xlabel_depo, ylabel=ylabel_depo, title=title_depo,
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
        plt.plot(time, sorc_Dp_plot, color='blue')
        axJ.set_xlim([0, T])
        axJ.set_ylim([0, 3500])
        axJ.set_xlabel('$t$ (hour)', fontsize=12)
        axJ.set_ylabel('$J(t)$ \n $(\mu$m$^{-1}$cm$^{-3}$ hour$^{-1}$)', fontsize=12, rotation=0)
        axJ.yaxis.set_label_coords(-0.015, 1.02)
        axJ.set_title('Nucleation rate', fontsize=12)
        axJ.grid()


    #######################################################
    # Images:

    # Parameters for size distribution images:
    yscale_image = 'linear'  # Change scale of y-axis (linear or log)
    xlabel_image = 'Time (hours)'  # x-axis label for image
    ylabel_image = '$D_p$ ($\mu$m)'  # y-axis label for image
    ylabelcoords = (-0.06, 1.05)  # y-axis label coordinates
    title_image = 'Size distribution'  # Title for image
    title_image_observations = 'Simulated observations'  # Title for image
    image_min = 100  # Minimum of image colour
    image_max = 12000  # Maximum of image colour
    cmap = 'jet'  # Colour map of image
    cbarlabel = '$\dfrac{dN}{dD_p}$ $(\mu$m$^{-1}$cm$^{-3})$'  # Label of colour bar
    cbarticks = [100, 1000, 10000]  # Ticks of colorbar

    # Plotting images:
    if plot_images:
        print('Plotting images...')
        basic_tools.image_plot(time, d_true, n_Dp_true, xlabel=xlabel_image, ylabel=ylabel_image, title=title_image,
                               yscale=yscale_image, ylabelcoords=ylabelcoords, image_min=image_min, image_max=image_max, cmap=cmap, cbarlabel=cbarlabel, cbarticks=cbarticks)
        basic_tools.image_plot(time, d_obs, Y, xlabel=xlabel_image, ylabel=ylabel_image, title=title_image_observations,
                               yscale=yscale_image, ylabelcoords=ylabelcoords, image_min=image_min, image_max=image_max, cmap=cmap, cbarlabel=cbarlabel, cbarticks=cbarticks)


    # Final print statements
    basic_tools.print_lines()  # Print lines in console
    print()  # Print space in console

"""

Title: Simualtes observations using the general dynamic equation of aerosols
Author: Vincent Russell
Date: June 22, 2022

"""


#######################################################
# Modules:
import numpy as np
import time as tm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tkinter import mainloop
from tqdm import tqdm

# Local modules:
import basic_tools
from observation_models.tools import get_DMA_transfer_function, compute_alpha_to_z_operator
from evolution_models.tools import GDE_evolution_model, GDE_Jacobian, change_basis_x_to_logDp, change_basis_x_to_logDp_sorc


#######################################################
# Importing parameter file:
from observation_models.simulators.observations_09.obs_09_parameters import *


#######################################################
if __name__ == '__main__':

    #######################################################
    # Initialising timer for total computation:
    initial_time = tm.time()  # Initial time stamp


    #######################################################
    # Constructing evolution model:
    F = GDE_evolution_model(Ne, Np, xmin, xmax, dt, NT, boundary_zero=boundary_zero, scale_type='log', print_status=False)  # Initialising evolution model
    F.add_process('condensation', cond_time(0))  # Adding condensation to evolution model
    F.add_process('deposition', depo_time(0))  # Adding deposition to evolution model
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
    def compute_evolution_operator(alpha_star, t_star, F, J_F):
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
    F_evolution, b_evolution = compute_evolution_operator(alpha[:, 0], t[0], F, J_F)  # Computing evolution operator from initial condition
    for k in tqdm(range(NT - 1)):  # Iterating over time
        F.add_process('condensation', cond_time(t[k]))  # Update condensation to evolution model
        F.add_process('deposition', depo_time(t[k]))  # Update condensation to evolution model
        J_F = GDE_Jacobian(F)  # Update Jacobian
        F_evolution, b_evolution = compute_evolution_operator(alpha[:, k], t[k], F, J_F)  # Computing evolution operator F and vector b
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
    channels = np.linspace(1, N_channels, N_channels)  # Array of integers up to number of channels
    if use_DMPS_observation_model:
        M = N_channels  # Setting M to number of channels (i.e. observation dimensions to number of channels)
        sample_volume = ((cpc_count_time / 60) * cpc_inlet_flow)  # Volume of aerosol sample (in litres)
        DMA_transfer_function = get_DMA_transfer_function(R_inner, R_outer, length, Q_aerosol, Q_sheath, efficiency)  # Computes DMA transfer function
        H_alpha_z = compute_alpha_to_z_operator(F, DMA_transfer_function, N_channels, voltage_min, voltage_max)  # Computes operator for computing z(t) given alpha(t)
        Y = np.zeros([N_channels, NT])  # Initialising observations
        for k in range(NT):
            z_k = np.matmul(H_alpha_z, alpha[:, k])  # Computing z_k
            Y[:, k] = (1 / sample_volume) * basic_tools.get_poisson(sample_volume * z_k)  # Drawing observations from Poisson distribution
    else:
        v_obs = diameter_to_volume(d_obs)  # Volumes that observations are made
        x_obs = np.log(v_obs)  # Log(volume) that observations are made
        _, _, n_x_obs, _ = F.get_nplot_discretisation(alpha, x_plot=x_obs)  # Computing plotting discretisation
        n_logDp_obs = change_basis_x_to_logDp(n_x_obs, v_obs, d_obs)  # Computing log_10(D_p)-based size distribution
        Y = (1 / sample_volume) * basic_tools.get_poisson(sample_volume * n_logDp_obs)  # Drawing observations from Poisson distribution

    Y += np.random.normal(additive_noise_mean, additive_noise_sigma, [N_channels, NT])  # Adding random noise
    Y[Y < 0] = 0  # Setting negative values to zero


    #######################################################
    # Computing parameters plotting discretisation:
    Nplot = len(d_true)  # Length of size discretisation
    cond_Dp_plot = np.zeros([Nplot, NT])  # Initialising ln(volume)-based condensation rate
    depo_plot = np.zeros([Nplot, NT])  # Initialising deposition rate
    sorc_x_plot = np.zeros(NT)  # Initialising ln(volume)-based source (nucleation) rate
    for k in range(NT):
        sorc_x_plot[k] = sorc(t[k])  # Computing ln(volume)-based nucleation rate
        for i in range(Nplot):
            cond_Dp_plot[i, k] = cond_time(t[k])(d_true[i])  # Computing ln(volume)-based condensation rate
            depo_plot[i, k] = depo_time(t[k])(d_true[i])  # Computing deposition rate
    sorc_logDp_plot = change_basis_x_to_logDp_sorc(sorc_x_plot, vmin, Dp_min)  # Computing log_10(D_p)-based nucleation rate


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
    xticks = [0.01, 0.1, 1]  # Plot x-tick labels
    xlimits = [d_true[0], d_true[-1]]  # Plot boundary limits for x-axis
    ylimits = [0, 10000]  # Plot boundary limits for y-axis
    xlabel = '$D_p$ ($\mu$m)'  # x-axis label for 1D animation plot
    ylabel = '$\dfrac{dN}{dlogD_p}$ (cm$^{-3})$'  # y-axis label for 1D animation plot
    title = 'Size distribution'  # Title for 1D animation plot
    legend = ['True size distribution', 'Simulated observations']  # Adding legend to plot
    line_color = ['blue', 'red']  # Colors of lines in plot
    time = t  # Array where time[i] is plotted (and animated)
    timetext = ('Time = ', ' hours')  # Tuple where text to be animated is: timetext[0] + 'time[i]' + timetext[1]
    delay = 30  # Delay between frames in milliseconds

    # Parameters for condensation plot:
    yscale_cond = 'linear'  # y-axis scaling ('linear' or 'log')
    ylimits_cond = [0, 0.04]  # Plot boundary limits for y-axis
    xlabel_cond = '$D_p$ ($\mu$m)'  # x-axis label for plot
    ylabel_cond = '$I(D_p)$ ($\mu$m hour$^{-1}$)'  # y-axis label for plot
    title_cond = 'Condensation rate'  # Title for plot
    location_cond = location + '2'  # Location for plot
    line_color_cond = ['blue']  # Colors of lines in plot

    # Parameters for deposition plot:
    yscale_depo = 'linear'  # y-axis scaling ('linear' or 'log')
    ylimits_depo = [0, 0.2]  # Plot boundary limits for y-axis
    xlabel_depo = '$D_p$ ($\mu$m)'  # x-axis label for plot
    ylabel_depo = '$d(D_p)$ (hour$^{-1}$)'  # y-axis label for plot
    title_depo = 'Deposition rate'  # Title for plot
    location_depo = location + '3'  # Location for plot
    line_color_depo = ['blue']  # Colors of lines in plot

    # Size distribution animation:
    if use_DMPS_observation_model:
        xlimits_SMPS = [1, N_channels]  # Plot boundary limits for x-axis
        ylimits_SMPS = [0, 500]  # Plot boundary limits for y-axis
        xlabel_SMPS = 'Channel'  # x-axis label for 1D animation plot
        ylabel_SMPS = 'Counts per litre'  # y-axis label for 1D animation plot
        title_SMPS = 'DMPS Observations'  # Title for 1D animation plot
        line_color_obs = ['red']  # Colors of lines in plot
        line_color_true = ['green']  # Colors of lines in plot
        basic_tools.plot_1D_animation(channels, Y, xlimits=xlimits_SMPS, ylimits=ylimits_SMPS, xlabel=xlabel_SMPS, ylabel=ylabel_SMPS, title=title_SMPS,
                                      delay=delay, location=location, time=time, timetext=timetext, line_color=line_color_obs, doing_mainloop=False)
        basic_tools.plot_1D_animation(d_true, n_logDp_true, xlimits=xlimits, ylimits=ylimits, xscale=xscale, xlabel=xlabel, ylabel=ylabel, title=title,
                                      delay=delay, location=location, time=time, timetext=timetext, line_color=line_color_true, doing_mainloop=False)
    else:
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
    # Plotting nucleation rate:
    if plot_nucleation:
        print('Plotting nucleation...')
        figJ, axJ = plt.subplots(figsize=(8.00, 5.00), dpi=100)
        plt.plot(time, sorc_logDp_plot, color='blue')
        axJ.set_xlim([0, T])
        axJ.set_ylim([0, 12000])
        axJ.set_xlabel('$t$ (hour)', fontsize=12)
        axJ.set_ylabel('$J(t)$ \n (cm$^{-3}$ hour$^{-1}$)', fontsize=12, rotation=0)
        axJ.yaxis.set_label_coords(-0.015, 1.02)
        axJ.set_title('Nucleation rate', fontsize=12)
        axJ.grid()


    #######################################################
    # Plotting DMA transfer functions
    if plot_dma_transfer_functions:
        N_plot = 1000
        DMA_transfer_function = get_DMA_transfer_function(R_inner, R_outer, length, Q_aerosol, Q_sheath, efficiency)  # Computes DMA transfer function
        N_voltages_plot = N_channels
        Dp = np.linspace(Dp_min, Dp_max, N_plot)
        voltage_plot = np.exp(np.linspace(np.log(voltage_min), np.log(voltage_max), N_voltages_plot))
        fig = plt.figure()
        # Setting axis variables:
        ax = fig.add_subplot(111)  # Creating subplot in figure
        ax.set_xlim(xlimits)  # Sets limits in x-axis
        ax.set_ylim([0, efficiency])  # Sets limits in y-axis
        ax.set_xscale(xscale)  # Sets x-axis to log or linear
        ax.set_xlabel(xlabel, fontsize=14)  # Adds xlabel
        ax.set_ylabel('$k_j(D_p)$', fontsize=14, rotation=0)  # Adds ylabel
        ax.yaxis.set_label_coords(-0.08, 1.02)
        ax.set_title('DMA Transfer Functions for channels $j = 1, \dots, M$', fontsize=14)  # Adds title
        plt.setp(ax, xticks=xticks, xticklabels=xticks)  # Modifies x-tick labels
        ax.grid()  # Adds a grid to the figure
        for j in range(N_voltages_plot):
            dma_plot = np.zeros(N_plot)  # Initialising
            for i in range(N_plot):
                dma_plot[i] = DMA_transfer_function(Dp[i], voltage_plot[j], 1)  # Computing
            plt.plot(Dp, dma_plot)


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

    # Parameters for condensation image:
    image_min_cond = 0.0003  # Minimum of image colour
    image_max_cond = 0.04  # Maximum of image colour
    cbarticks_cond = [0.001, 0.01]  # Ticks of colorbar

    # Parameters for deposition image:
    image_min_depo = 0.005  # Minimum of image colour
    image_max_depo = 0.15  # Maximum of image colour
    cbarticks_depo = [0.01, 0.1]  # Ticks of colorbar

    # Plotting images:
    if plot_images:
        print('Plotting images...')
        basic_tools.image_plot(time, d_true, n_logDp_true, xlabel=xlabel_image, ylabel=ylabel_image, title=title_image,
                               yscale=yscale_image, ylabelcoords=ylabelcoords, yticks=yticks_image, image_min=image_min, image_max=image_max, cmap=cmap, cbarlabel=cbarlabel, cbarticks=cbarticks)
        basic_tools.image_plot(time, d_true, cond_Dp_plot, xlabel=xlabel_image, ylabel=ylabel_image, title=title_cond,
                               yscale=yscale_image, ylabelcoords=ylabelcoords, yticks=yticks_image, image_min=image_min_cond, image_max=image_max_cond, cmap=cmap, cbarlabel=ylabel_cond, cbarticks=cbarticks_cond)
        basic_tools.image_plot(time, d_true, depo_plot, xlabel=xlabel_image, ylabel=ylabel_image, title=title_depo,
                               yscale=yscale_image, ylabelcoords=ylabelcoords, yticks=yticks_image, image_min=image_min_depo, image_max=image_max_depo, cmap=cmap, cbarlabel=ylabel_depo, cbarticks=cbarticks_depo)


    # Final print statements
    basic_tools.print_lines()  # Print lines in console
    print()  # Print space in console


    #######################################################
    # Temporary Plotting:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "DejaVu Sans",
    })

    # Parameters:
    image_min = 10
    image_max = 10000
    cbarticks = [10, 100, 1000, 10000]
    image_min_obs = 1
    image_max_obs = 500
    cbarticks_obs = [1, 10, 100, 500]

    # fig image:
    fig5, ax = plt.subplots(figsize=(8, 4), dpi=200)
    n_logDp_true = n_logDp_true.clip(image_min, image_max)
    im = plt.pcolor(time, d_true, n_logDp_true, cmap=cmap, vmin=image_min, vmax=image_max, norm=LogNorm())
    cbar = fig5.colorbar(im, ticks=cbarticks, orientation='vertical')
    tick_labels = [str(tick) for tick in cbarticks]
    cbar.ax.set_yticklabels(tick_labels)
    cbar.set_label(r'$\displaystyle\frac{dN}{dlogD_p}$ (cm$^{-3})$', fontsize=12, rotation=0, y=1.2, labelpad=-10)
    ax.set_xlabel('Time (hours)', fontsize=14)
    ax.set_ylabel(xlabel, fontsize=14, rotation=0)
    ax.yaxis.set_label_coords(-0.05, 1.05)
    ax.set_title('True size distribution', fontsize=14)
    ax.set_xlim([0, T])
    ax.set_ylim([Dp_min, Dp_max])
    ax.set_yscale('log')
    plt.setp(ax, yticks=xticks, yticklabels=xticks)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    plt.tight_layout()
    fig5.savefig('image_truth')

    # fig image observations:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
    Y = Y.clip(image_min_obs, image_max_obs)
    im = plt.pcolor(time, channels, Y, cmap=cmap, vmin=image_min_obs, vmax=image_max_obs, norm=LogNorm())
    cbar = fig.colorbar(im, ticks=cbarticks_obs, orientation='vertical')
    tick_labels = [str(tick) for tick in cbarticks_obs]
    cbar.ax.set_yticklabels(tick_labels)
    cbar.set_label('Counts per litre', fontsize=12, rotation=0, y=1.1, labelpad=-30)
    ax.set_xlabel('Time (hours)', fontsize=14)
    ax.set_ylabel('Channel', fontsize=14, rotation=0)
    ax.yaxis.set_label_coords(-0.05, 1.05)
    ax.set_title('Simulated observations', fontsize=14)
    ax.set_xlim([0, T])
    ax.set_ylim([1, N_channels])
    ax.set_yscale('linear')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    plt.tight_layout()
    fig.savefig('image_observations')
"""

Title: State estimation of aerosol particle size distribution
Author: Vincent Russell
Date: June 27, 2022

"""


#######################################################
# Modules:
import numpy as np
import time as tm
from tkinter import mainloop
from tqdm import tqdm

# Local modules:
import basic_tools
from basic_tools import Kalman_filter, compute_fixed_interval_Kalman_smoother
from observation_models.data.simulated import load_observations
from evolution_models.tools import GDE_evolution_model, GDE_Jacobian, change_basis_x_to_logDp
from observation_models.tools import Size_distribution_observation_model


#######################################################
# Importing parameter file:
from state_estimation_case_studies.case_study_15.state_est_15_parameters import *


#######################################################
if __name__ == '__main__':

    #######################################################
    # Initialising timer for total computation:
    initial_time = tm.time()  # Initial time stamp


    #######################################################
    # Importing simulated observations and true size distribution:
    observation_data = load_observations(data_filename)  # Loading data file
    d_obs, Y = observation_data['d_obs'], observation_data['Y']  # Extracting observations
    d_true, n_x_true = observation_data['d_true'], observation_data['n_true']  # Extracting true size distribution
    n_logDp_true = change_basis_x_to_logDp(n_x_true, diameter_to_volume(d_true), d_true)  # Computing diameter-based size distribution


    #######################################################
    # Constructing size distribution evolution model:
    F_alpha = GDE_evolution_model(Ne, Np, xmin, xmax, dt, NT, boundary_zero=boundary_zero, scale_type='log')  # Initialising evolution model
    F_alpha.add_process('condensation', guess_cond)  # Adding condensation to evolution model
    F_alpha.add_process('deposition', guess_depo)  # Adding deposition to evolution model
    F_alpha.add_process('coagulation', coag, load_coagulation=load_coagulation, coagulation_suffix=coagulation_suffix)  # Adding coagulation to evolution model
    F_alpha.compile()  # Compiling evolution model


    #######################################################
    # Constructing Jacobian of size distribution evolution model:
    J_F_alpha = GDE_Jacobian(F_alpha)  # Evolution Jacobian
    dF_alpha_d_alpha = J_F_alpha.eval_d_alpha  # Derivative with respect to alpha


    #######################################################
    # Functions to compute/update evolution operator, Jacobians, and covariance using Crank-Nicolson method:

    # Function to compute evolution operator Jacobians:
    def compute_evolution_operator_Jacobians(alpha_star, t_star):
        # Computing Jacobians:
        J_alpha_star = dF_alpha_d_alpha(alpha_star, t_star)
        return J_alpha_star

    # Function to compute evolution operator:
    def compute_evolution_operator(alpha_star, t_star, J_alpha_star):
        # Computing F_star:
        F_star = F_alpha.eval(alpha_star, t_star) - np.matmul(J_alpha_star, alpha_star)
        # Computing evolution operators for each coefficient:
        matrix_multiplier = np.linalg.inv(np.eye(N) - (dt / 2) * J_alpha_star)  # Computing matrix multiplier for evolution operators and additive vector
        F_evol_alpha = np.matmul(matrix_multiplier, (np.eye(N) + (dt / 2) * J_alpha_star))
        # Computing evolution operator:
        F_evolution = np.zeros([N, N])  # Initialising
        F_evolution[0:N, 0:N] = F_evol_alpha
        # Computing evolution additive vector:
        b_evolution = np.zeros(N)  # Initialising
        b_evolution[0:N] = np.matmul(matrix_multiplier, (dt * F_star))
        return F_evolution, b_evolution


    #######################################################
    # Constructing observation model:
    M = len(Y)  # Dimension size of observations
    H_alpha = Size_distribution_observation_model(F_alpha, d_obs, M)  # Observation model
    H = np.zeros([M, N])  # Initialising
    H[0:M, 0:N] = H_alpha.H_phi  # Observation operator


    #######################################################
    # Constructing noise covariance for observation model:
    Gamma_v = np.zeros([NT, M, M])  # Initialising
    for k in range(NT):
        Gamma_Y_multiplier = (sigma_Y_multiplier ** 2) * np.diag(Y[:, k])  # Noise proportional to Y
        Gamma_v_additive = (sigma_v ** 2) * np.eye(M)  # Additive noise
        Gamma_v[k] = Gamma_Y_multiplier + Gamma_v_additive  # Observation noise covariance


    #######################################################
    # Constructing noise covariances for evolution model:

    # For alpha:
    sigma_alpha_w = np.array([sigma_alpha_w_0, sigma_alpha_w_1, sigma_alpha_w_2, sigma_alpha_w_3, sigma_alpha_w_4, sigma_alpha_w_5, sigma_alpha_w_6])  # Array of standard deviations
    Gamma_alpha_w = basic_tools.compute_correlated_covariance_matrix(N, Np, Ne, sigma_alpha_w, sigma_alpha_w_correlation)  # Covariance matrix computation
    Gamma_alpha_w[0: Np, 0: Np] = alpha_first_element_multiplier * Gamma_alpha_w[0: Np, 0: Np]  # First element multiplier

    # Assimilation:
    Gamma_w = np.zeros([N, N])  # Initialising noise covariance for state
    Gamma_w[0:N, 0:N] = Gamma_alpha_w  # Adding alpha covariance to state covariance


    #######################################################
    # Constructing prior and prior covariance:

    # For alpha:
    alpha_prior = F_alpha.compute_coefficients('alpha', initial_guess_size_distribution)  # Computing alpha coefficients from initial condition function
    sigma_alpha_prior = np.array([sigma_alpha_prior_0, sigma_alpha_prior_1, sigma_alpha_prior_2, sigma_alpha_prior_3, sigma_alpha_prior_4, sigma_alpha_prior_5, sigma_alpha_prior_6])  # Array of standard deviations
    Gamma_alpha_prior = basic_tools.compute_correlated_covariance_matrix(N, Np, Ne, sigma_alpha_prior, 0.001)  # Covariance matrix computation
    Gamma_alpha_prior[0: Np, 0: Np] = alpha_first_element_multiplier * Gamma_alpha_prior[0: Np, 0: Np]  # First element multiplier

    # Assimilation:
    x_prior = np.zeros([N])  # Initialising prior state
    x_prior[0:N] = alpha_prior  # Adding alpha prior to prior state
    Gamma_prior = np.zeros([N, N])  # Initialising prior covariance for state
    Gamma_prior[0:N, 0:N] = Gamma_alpha_prior  # Adding alpha covariance to state covariance


    #######################################################
    # Initialising state and adding prior:
    x = np.zeros([N, NT])  # Initialising state = [x_0, x_1, ..., x_{NT - 1}]
    Gamma = np.zeros([NT, N, N])  # Initialising state covariance matrix Gamma_0, Gamma_1, ..., Gamma_NT
    x_predict = np.zeros([N, NT])  # Initialising predicted state
    Gamma_predict = np.zeros([NT, N, N])  # Initialising predicted state covariance
    x[:, 0], x_predict[:, 0] = x_prior, x_prior  # Adding prior to states
    Gamma[0], Gamma_predict[0] = Gamma_prior, Gamma_prior  # Adding prior to state covariance matrices


    #######################################################
    # Initialising evolution operator and additive evolution vector:
    F = np.zeros([NT, N, N])  # Initialising evolution operator F_0, F_1, ..., F_{NT - 1}
    b = np.zeros([N, NT])  # Initialising additive evolution vector b_0, b_1, ..., b_{NT - 1}


    #######################################################
    # Constructing extended Kalman filter model:
    model = Kalman_filter(F[0], H, Gamma_w, Gamma_v, NT, additive_evolution_vector=b[:, 0])


    #######################################################
    # Computing time evolution of model:
    print('Computing Kalman filter estimates...')
    t = np.zeros(NT)  # Initialising time array
    for k in tqdm(range(NT - 1)):  # Iterating over time
        J_alpha_star = compute_evolution_operator_Jacobians(x[:, k], t[k])  # Computing evolution operator Jacobian
        F[k], b[:, k] = compute_evolution_operator(x[:, k], t[k], J_alpha_star)  # Computing evolution operator F and vector b
        model.F, model.additive_evolution_vector = F[k], b[:, k]  # Adding updated evolution operator and vector b to Kalman Filter
        x_predict[:, k + 1], Gamma_predict[k + 1] = model.predict(x[:, k], Gamma[k], k)  # Computing prediction
        x[:, k + 1], Gamma[k + 1] = model.update(x_predict[:, k + 1], Gamma_predict[k + 1], Y[:, k + 1], k)  # Computing update
        t[k + 1] = (k + 1) * dt  # Time (hours)


    #######################################################
    # Computing smoothed estimates:
    if smoothing:
        print('Computing Kalman smoother estimates...')
        x, Gamma = compute_fixed_interval_Kalman_smoother(F, NT, N, x, Gamma, x_predict, Gamma_predict)


    #######################################################
    # Extracting alpha from state:
    alpha = x[0:N, :]  # Size distribution coefficients
    Gamma_alpha = Gamma[:, 0:N, 0:N]  # Size distribution covariance


    #######################################################
    # Computing plotting discretisation:
    # Size distribution:
    d_plot, v_plot, n_logDp_plot, sigma_n_logDp = F_alpha.get_nplot_discretisation(alpha, Gamma_alpha=Gamma_alpha, convert_x_to_logDp=True)
    n_Dp_plot_upper = n_logDp_plot + 2 * sigma_n_logDp
    n_Dp_plot_lower = n_logDp_plot - 2 * sigma_n_logDp


    #######################################################
    # Computing true and guessed underlying parameters plotting discretisation:
    Nplot_cond = len(d_plot)  # Length of size discretisation
    Nplot_depo = len(d_plot)  # Length of size discretisation
    cond_Dp_truth_plot = np.zeros([Nplot_cond, NT])  # Initialising condensation rate
    cond_Dp_guess_plot = np.zeros([Nplot_cond, NT])  # Initialising condensation rate
    depo_truth_plot = np.zeros([Nplot_depo, NT])  # Initialising deposition rate
    depo_guess_plot = np.zeros([Nplot_depo, NT])  # Initialising deposition rate
    for k in range(NT):
        for i in range(Nplot_cond):
            cond_Dp_truth_plot[i, k] = cond(d_plot[i])  # Computing condensation rate
            cond_Dp_guess_plot[i, k] = guess_cond(d_plot[i])  # Computing condensation rate
        for i in range(Nplot_depo):
            depo_truth_plot[i, k] = depo(d_plot[i])  # Computing deposition rate
            depo_guess_plot[i, k] = guess_depo(d_plot[i])  # Computing deposition rate


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
    xticks = [Dp_min, 0.1, Dp_max]  # Plot x-tick labels
    xlimits = [Dp_min, Dp_max]  # Plot boundary limits for x-axis
    ylimits = [0, 300]  # Plot boundary limits for y-axis
    xlabel = '$D_p$ ($\mu$m)'  # x-axis label for 1D animation plot
    ylabel = '$\dfrac{dN}{dlogD_p}$ (cm$^{-3})$'  # y-axis label for 1D animation plot
    title = 'Size distribution estimation'  # Title for 1D animation plot
    legend = ['Estimate', '$\pm 2 \sigma$', '', 'Truth']  # Adding legend to plot
    line_color = ['blue', 'blue', 'blue', 'green']  # Colors of lines in plot
    line_style = ['solid', 'dashed', 'dashed', 'solid']  # Style of lines in plot
    time = t  # Array where time[i] is plotted (and animated)
    timetext = ('Time = ', ' hours')  # Tuple where text to be animated is: timetext[0] + 'time[i]' + timetext[1]
    delay = 60  # Delay between frames in milliseconds

    # Parameters for condensation plot:
    yscale_cond = 'linear'  # y-axis scaling ('linear' or 'log')
    ylimits_cond = [0, 0.01]  # Plot boundary limits for y-axis
    xlabel_cond = '$D_p$ ($\mu$m)'  # x-axis label for plot
    ylabel_cond = '$I(D_p)$ ($\mu$m hour$^{-1}$)'  # y-axis label for plot
    title_cond = 'Condensation rate estimation'  # Title for plot
    location_cond = location + '2'  # Location for plot
    legend_cond = ['Guess', 'Truth']  # Adding legend to plot
    line_color_cond = ['blue', 'green']  # Colors of lines in plot
    line_style_cond = ['solid', 'solid']  # Style of lines in plot
    delay_cond = 30  # Delay for each frame (ms)

    # Parameters for deposition plot:
    yscale_depo = 'linear'  # y-axis scaling ('linear' or 'log')
    ylimits_depo = [0, 1]  # Plot boundary limits for y-axis
    xlabel_depo = '$D_p$ ($\mu$m)'  # x-axis label for plot
    ylabel_depo = '$d(D_p)$ (hour$^{-1}$)'  # y-axis label for plot
    title_depo = 'Deposition rate estimation'  # Title for plot
    location_depo = location + '3'  # Location for plot
    legend_depo = ['Guess', 'Truth']  # Adding legend to plot
    line_color_depo = ['blue', 'green']  # Colors of lines in plot
    line_style_depo = ['solid', 'solid']  # Style of lines in plot
    delay_depo = delay_cond  # Delay between frames in milliseconds

    # Size distribution animation:
    basic_tools.plot_1D_animation(d_plot, n_logDp_plot, n_Dp_plot_lower, n_Dp_plot_upper, plot_add=(d_true, n_logDp_true), xticks=xticks, xlimits=xlimits, ylimits=ylimits, xscale=xscale, xlabel=xlabel, ylabel=ylabel, title=title,
                                  delay=delay, location=location, legend=legend, time=time, timetext=timetext, line_color=line_color, line_style=line_style, doing_mainloop=False)

    # Condensation rate animation:
    basic_tools.plot_1D_animation(d_plot, cond_Dp_guess_plot, cond_Dp_truth_plot, xticks=xticks, xlimits=xlimits, ylimits=ylimits_cond, xscale=xscale, yscale=yscale_cond, xlabel=xlabel_cond, ylabel=ylabel_cond, title=title_cond,
                                  location=location_cond, delay=delay_cond, legend=legend_cond, time=time, timetext=timetext, line_color=line_color_cond, line_style=line_style_cond, doing_mainloop=False)

    # Deposition rate animation:
    basic_tools.plot_1D_animation(d_plot, depo_guess_plot, depo_truth_plot, xticks=xticks, xlimits=xlimits, ylimits=ylimits_depo, xscale=xscale, yscale=yscale_depo, xlabel=xlabel_depo, ylabel=ylabel_depo, title=title_depo,
                                  location=location_depo, delay=delay_depo, legend=legend_depo, time=time, timetext=timetext, line_color=line_color_depo, line_style=line_style_depo, doing_mainloop=False)

    # Mainloop and print:
    if plot_animations:
        print('Plotting animations...')
        mainloop()  # Runs tkinter GUI for plots and animations


    #######################################################
    # Images:

    # Parameters for size distribution images:
    yscale_image = 'log'  # Change scale of y-axis (linear or log)
    yticks_image = [Dp_min, 0.1, Dp_max]  # Plot y-tick labels
    xlabel_image = 'Time (hours)'  # x-axis label for image
    ylabel_image = '$D_p$ ($\mu$m)'  # y-axis label for image
    ylabelcoords = (-0.06, 1.05)  # y-axis label coordinates
    title_image = 'Size distribution estimation'  # Title for image
    title_image_observations = 'Simulated observations'  # Title for image
    image_min = 1  # Minimum of image colour
    image_max = 300  # Maximum of image colour
    cmap = 'jet'  # Colour map of image
    cbarlabel = '$\dfrac{dN}{dlogD_p}$ (cm$^{-3})$'  # Label of colour bar
    cbarticks = [1, 10, 100]  # Ticks of colorbar

    # Plotting images:
    if plot_images:
        print('Plotting images...')
        basic_tools.image_plot(time, d_plot, n_logDp_plot, xlabel=xlabel_image, ylabel=ylabel_image, title=title_image,
                               yscale=yscale_image, ylabelcoords=ylabelcoords, yticks=yticks_image, image_min=image_min, image_max=image_max, cmap=cmap, cbarlabel=cbarlabel, cbarticks=cbarticks)
        basic_tools.image_plot(time, d_obs, Y, xlabel=xlabel_image, ylabel=ylabel_image, title=title_image_observations,
                               yscale=yscale_image, ylabelcoords=ylabelcoords, yticks=yticks_image, image_min=image_min, image_max=image_max, cmap=cmap, cbarlabel=cbarlabel, cbarticks=cbarticks)


    # Final print statements
    basic_tools.print_lines()  # Print lines in console
    print()  # Print space in console

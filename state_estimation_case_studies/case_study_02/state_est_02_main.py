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
from basic_tools import Extended_Kalman_filter, compute_fixed_interval_extended_Kalman_smoother
from observation_models.data.simulated import load_observations
from evolution_models.tools import GDE_evolution_model, GDE_Jacobian
from observation_models.tools import Size_distribution_observation_model, Size_distribution_observation_model_Jacobian


#######################################################
# Importing parameter file:
from state_estimation_case_studies.case_study_02.state_est_02_parameters import *


#######################################################
if __name__ == '__main__':

    #######################################################
    # Initialising timer for total computation:
    initial_time = tm.time()  # Initial time stamp


    #######################################################
    # Importing simulated observations and true size distribution:
    observation_data = load_observations(data_filename)  # Loading data file
    d_obs, Y = observation_data['d_obs'], observation_data['Y']  # Extracting observations
    # d_true, n_v_true = observation_data['d_true'], observation_data['n_true']  # Extracting true size distribution
    # v_obs = basic_tools.diameter_to_volume(d_obs)  # Converting diameter observations to volume
    # v_true = basic_tools.diameter_to_volume(d_true)  # Converting true diameter to volume


    #######################################################
    # Constructing size distribution evolution model:
    F_alpha = GDE_evolution_model(Ne, Np, vmin, vmax, dt, NT, boundary_zero=boundary_zero)  # Initialising evolution model
    F_alpha.add_process('condensation', guess_cond)  # Adding condensation to evolution model
    F_alpha.add_process('deposition', guess_depo)  # Adding deposition to evolution model
    F_alpha.add_process('source', guess_sorc)  # Adding deposition to evolution model
    F_alpha.add_process('coagulation', coag, load_coagulation=load_coagulation, coagulation_suffix=coagulation_suffix)  # Adding coagulation to evolution model
    F_alpha.compile(time_integrator='euler')  # Compiling evolution model and adding time integrator


    #######################################################
    # Constructing state evolution model:
    def F(alpha, t):
        return F_alpha.eval(alpha, t)


    #######################################################
    # Constructing Jacobian of size distribution evolution model:
    J_F_alpha = GDE_Jacobian(F_alpha)  # Evolution Jacobian
    dF_alpha_d_alpha = J_F_alpha.eval_d_alpha  # Derivative with respect to alpha


    #######################################################
    # Constructing Jacobian of state evolution model:
    def J_F(alpha, t):
        return dF_alpha_d_alpha(alpha, t)


    #######################################################
    # Constructing observation model:
    M = len(Y)  # Dimension size of observations
    H_alpha = Size_distribution_observation_model(F_alpha, d_obs, M)  # Observation model
    def H(x, *_):
        alpha = x[0: N]  # Extracting alpha from state x
        return H_alpha.eval(alpha)


    #######################################################
    # Constructing Jacobian of observation model:
    J_H_alpha = Size_distribution_observation_model_Jacobian(H_alpha)  # Observation Jacobian
    def J_H(*_):
        output = np.zeros([M, N])
        output[0:M, 0:N] = J_H_alpha.eval()
        return output


    #######################################################
    # Constructing noise covariance for observation model:
    Gamma_v = np.zeros([NT, M, M])  # Initialising
    for t in range(NT):
        Gamma_Y_multiplier = (sigma_Y_multiplier ** 2) * np.diag(Y[:, t])  # Noise proportional to Y
        Gamma_v_additive = (sigma_v ** 2) * np.eye(M)  # Additive noise
        Gamma_v[t] = Gamma_Y_multiplier + Gamma_v_additive  # Observation noise covariance


    #######################################################
    # Constructing noise covariances for evolution model:

    # For alpha:
    sigma_alpha_w = np.array([sigma_alpha_w_0, sigma_alpha_w_1, sigma_alpha_w_2, sigma_alpha_w_3, sigma_alpha_w_4, sigma_alpha_w_5, sigma_alpha_w_6])  # Array of standard deviations
    Gamma_alpha_w = basic_tools.compute_correlated_covariance_matrix(N, Np, Ne, sigma_alpha_w, sigma_alpha_w_correlation)  # Covariance matrix computation

    # Assimilation:
    Gamma_w = np.zeros([N, N])  # Initialising noise covariance for state
    Gamma_w[0:N, 0:N] = Gamma_alpha_w  # Adding alpha covariance to state covariance


    #######################################################
    # Constructing prior and prior covariance:

    # For alpha:
    alpha_prior = F_alpha.compute_coefficients('alpha', initial_guess_size_distribution)  # Computing alpha coefficients from initial condition function
    sigma_alpha_prior = np.array([sigma_alpha_prior_0, sigma_alpha_prior_1, sigma_alpha_prior_2, sigma_alpha_prior_3, sigma_alpha_prior_4, sigma_alpha_prior_5, sigma_alpha_prior_6])  # Array of standard deviations
    Gamma_alpha_prior = basic_tools.compute_correlated_covariance_matrix(N, Np, Ne, sigma_alpha_prior, 0.001)  # Covariance matrix computation

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
    # Constructing extended Kalman filter model:
    model = Extended_Kalman_filter(F, J_F, H, J_H, Gamma_w, Gamma_v, NT)


    #######################################################
    # Computing time evolution of model:
    print('Computing Extended Kalman filter estimates...')
    t = np.zeros(NT)  # Initialising time array
    for k in tqdm(range(NT - 1)):  # Iterating over time
        x_predict[:, k + 1], Gamma_predict[k + 1] = model.predict(x[:, k], Gamma[k], t[k], k)  # Computing prediction
        x[:, k + 1], Gamma[k + 1],  = model.update(x_predict[:, k + 1], Gamma_predict[k + 1], Y[:, k + 1], t[k], k)  # Computing update
        t[k + 1] = (k + 1) * dt  # Time (hours)


    #######################################################
    # Computing smoothed estimates:
    if smoothing:
        print('Computing Extended Kalman smoother estimates...')
        x, Gamma = compute_fixed_interval_extended_Kalman_smoother(J_F, dt, NT, N, x, Gamma, x_predict, Gamma_predict)


    #######################################################
    # Extracting alpha, gamma, eta, and covariances from state:
    alpha = x[0:N, :]  # Size distribution coefficients
    Gamma_alpha = Gamma[:, 0:N, 0:N]  # Size distribution covariance


    #######################################################
    # Computing plotting discretisation:
    # Size distribution:
    d_plot, v_plot, n_Dp_plot, sigma_n_Dp = F_alpha.get_nplot_discretisation(alpha, Gamma_alpha=Gamma_alpha, convert_v_to_Dp=True)
    n_Dp_plot_upper = n_Dp_plot + 2 * sigma_n_Dp
    n_Dp_plot_lower = n_Dp_plot - 2 * sigma_n_Dp


    #######################################################
    # Computing true underlying parameters plotting discretisation:
    Nplot_cond = len(d_plot)  # Length of size discretisation
    Nplot_depo = len(d_plot)  # Length of size discretisation
    cond_Dp_true_plot = np.zeros([Nplot_cond, NT])  # Initialising volume-based condensation rate
    depo_true_plot = np.zeros([Nplot_depo, NT])  # Initialising deposition rate
    for k in range(NT):
        for i in range(Nplot_cond):
            cond_Dp_true_plot[i, k] = cond(d_plot[i])  # Computing volume-based condensation rate
        for i in range(Nplot_depo):
            depo_true_plot[i, k] = depo(d_plot[i])  # Computing deposition rate


    #######################################################
    # Printing total computation time:
    computation_time = round(tm.time() - initial_time, 3)  # Initial time stamp
    print('Total computation time:', str(computation_time), 'seconds.')  # Print statement


    #######################################################
    # Plotting size distribution animation and function parameters:

    # General parameters:
    location = 'Home'  # Set to 'Uni', 'Home', or 'Middle' (default)

    # Parameters for size distribution animation:
    xscale = 'linear'  # x-axis scaling ('linear' or 'log')
    xlimits = [d_plot[0], d_plot[-1]]  # Plot boundary limits for x-axis
    ylimits = [0, 10000]  # Plot boundary limits for y-axis
    xlabel = '$D_p$ ($\mu$m)'  # x-axis label for 1D animation plot
    ylabel = '$\dfrac{dN}{dD_p}$ $(\mu$m$^{-1}$cm$^{-3})$'  # y-axis label for 1D animation plot
    title = 'Size distribution estimation'  # Title for 1D animation plot
    legend = ['Estimate', '$\pm 2 \sigma$', '', 'Simulated observations']  # Adding legend to plot
    line_color = ['blue', 'blue', 'blue', 'red']  # Colors of lines in plot
    line_style = ['solid', 'dashed', 'dashed', 'solid']  # Style of lines in plot
    time = t  # Array where time[i] is plotted (and animated)
    timetext = ('Time = ', ' hours')  # Tuple where text to be animated is: timetext[0] + 'time[i]' + timetext[1]

    # Parameters for condensation plot:
    ylimits_cond = [0, 2]  # Plot boundary limits for y-axis
    xlabel_cond = '$D_p$ ($\mu$m)'  # x-axis label for plot
    ylabel_cond = '$I(D_p)$ ($\mu$m hour$^{-1}$)'  # y-axis label for plot
    title_cond = 'Condensation rate estimation'  # Title for plot
    location_cond = location + '2'  # Location for plot
    legend_cond = ['Truth']  # Adding legend to plot
    line_color_cond = ['green']  # Colors of lines in plot
    line_style_cond = ['solid']  # Style of lines in plot
    delay_cond = 0  # Delay for each frame (ms)

    # Parameters for deposition plot:
    ylimits_depo = [0, 0.6]  # Plot boundary limits for y-axis
    xlabel_depo = '$D_p$ ($\mu$m)'  # x-axis label for plot
    ylabel_depo = '$d(D_p)$ (hour$^{-1}$)'  # y-axis label for plot
    title_depo = 'Deposition rate estimation'  # Title for plot
    location_depo = location + '3'  # Location for plot
    legend_depo = ['Truth']  # Adding legend to plot
    line_color_depo = ['green']  # Colors of lines in plot
    line_style_depo = ['solid']  # Style of lines in plot
    delay_depo = delay_cond  # Delay between frames in milliseconds

    # Size distribution animation:
    basic_tools.plot_1D_animation(d_plot, n_Dp_plot, n_Dp_plot_lower, n_Dp_plot_upper, plot_add=(d_obs, Y), xlimits=xlimits, ylimits=ylimits, xscale=xscale, xlabel=xlabel, ylabel=ylabel, title=title,
                                  location=location, legend=legend, time=time, timetext=timetext, line_color=line_color, line_style=line_style, doing_mainloop=False)

    # Condensation rate animation:
    basic_tools.plot_1D_animation(d_plot, cond_Dp_true_plot, xlimits=xlimits, ylimits=ylimits_cond, xscale=xscale, xlabel=xlabel_cond, ylabel=ylabel_cond, title=title_cond,
                                  location=location_cond, delay=delay_cond, legend=legend_cond, time=time, timetext=timetext, line_color=line_color_cond, line_style=line_style_cond, doing_mainloop=False)

    # Deposition rate animation:
    basic_tools.plot_1D_animation(d_plot, depo_true_plot, xlimits=xlimits, ylimits=ylimits_depo, xscale=xscale, xlabel=xlabel_depo, ylabel=ylabel_depo, title=title_depo,
                                  location=location_depo, delay=delay_depo, legend=legend_depo, time=time, timetext=timetext, line_color=line_color_depo, line_style=line_style_depo, doing_mainloop=False)

    # Mainloop and print:
    if plot_animations:
        print('Plotting animations...')
        mainloop()  # Runs tkinter GUI for plots and animations


    #######################################################
    # Images:

    # Parameters for size distribution images:
    yscale_image = 'linear'  # Change scale of y-axis (linear or log)
    xlabel_image = 'Time (hours)'  # x-axis label for image
    ylabel_image = '$D_p$ ($\mu$m)'  # y-axis label for image
    ylabelcoords = (-0.06, 1.05)  # y-axis label coordinates
    title_image = 'Size distribution estimation'  # Title for image
    title_image_observations = 'Simulated observations'  # Title for image
    image_min = 100  # Minimum of image colour
    image_max = 10000  # Maximum of image colour
    cmap = 'jet'  # Colour map of image
    cbarlabel = '$\dfrac{dN}{dD_p}$ $(\mu$m$^{-1}$cm$^{-3})$'  # Label of colour bar
    cbarticks = [100, 1000, 10000]  # Ticks of colorbar

    # Plotting images:
    if plot_images:
        print('Plotting images...')
        basic_tools.image_plot(time, d_plot, n_Dp_plot, xlabel=xlabel_image, ylabel=ylabel_image, title=title_image,
                               yscale=yscale_image, ylabelcoords=ylabelcoords, image_min=image_min, image_max=image_max, cmap=cmap, cbarlabel=cbarlabel, cbarticks=cbarticks)
        basic_tools.image_plot(time, d_obs, Y, xlabel=xlabel_image, ylabel=ylabel_image, title=title_image_observations,
                               yscale=yscale_image, ylabelcoords=ylabelcoords, image_min=image_min, image_max=image_max, cmap=cmap, cbarlabel=cbarlabel, cbarticks=cbarticks)


    # Final print statements
    basic_tools.print_lines()  # Print lines in console
    print()  # Print space in console

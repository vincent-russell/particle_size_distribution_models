"""

Title: State estimation of wave simulated by the advection equation
Author: Vincent Russell
Date: August 31, 2021

"""


#######################################################
# Modules:
import numpy as np
import pandas as pd
import time as tm
import matplotlib.pyplot as plt
from tkinter import mainloop

# Local modules:
import basic_tools
import basic_models.advection_models.tools as tools
from state_space_identification_models.algorithms import Kalman_filter, compute_fixed_interval_Kalman_smoother


#######################################################
if __name__ == '__main__':

    #######################################################
    # Fixed parameters:

    # Setup and plotting:
    smoothing = False  # Set to True to compute fixed interval Kalman smoother estimates
    plot_animations = True  # Set to True to plot animations
    use_BAE = True  # Set to True to use BAE in state estimation
    compute_weighted_norm = True  # Sest to True to compute weighted norm difference (weighted by inverse of sigma_n)
    observation_number = 1  # Can get specific observation number, or set to None for random

    # Spatial domain:
    xmin = 0  # Minimum
    xmax = 1  # Maximum

    # Time domain:
    dt = 0.001  # Time step
    T = 1  # End time
    NT = int(T / dt)  # Total number of time steps

    # Solution discretisation:
    Ne = 10  # Number of elements
    Np = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
    N = Ne * Np  # Total degrees of freedom


    #######################################################
    # Guess parameters:

    # Prior noise parameters:
    # Prior covariance for alpha; Gamma_alpha_prior = sigma_alpha_prior^2 * I_N:
    sigma_alpha_prior_0 = 0.001
    sigma_alpha_prior_1 = sigma_alpha_prior_0 / 2
    sigma_alpha_prior_2 = sigma_alpha_prior_1 / 4

    # Model noise parameters:
    # Observation noise covariance Gamma_v = sigma_v^2 * I_N:
    sigma_v = 0.2
    # Evolution noise covariance Gamma_alpha_w = sigma_alpha_w^2 * I_N:
    sigma_alpha_w_0 = 0.001
    sigma_alpha_w_1 = sigma_alpha_w_0 / 2
    sigma_alpha_w_2 = sigma_alpha_w_1 / 4
    sigma_alpha_w_correlation = 1

    # Initial condition n_0(x) = n(x, 0):
    N_0_guess = 1  # Amplitude of initial condition gaussian
    mu_0_guess = 0.2  # Mean of initial condition gaussian
    sigma_0_guess = 0.04  # Standard deviation of initial condition gaussian
    def initial_condition(x):
        return basic_tools.gaussian(x, N_0_guess, mu_0_guess, sigma_0_guess)

    # Advection model:
    c_0_guess = 0.4  # Constant coefficient
    c_1_guess = 0  # Linear coefficient
    c_2_guess = 0  # Quadratic coefficient
    def advection(x):
        return c_0_guess + c_1_guess * x + c_2_guess * x ** 2


    #######################################################
    # Initialising timer for total computation:
    basic_tools.print_lines()
    initial_time = tm.time()  # Initial time stamp


    #######################################################
    # Importing simulated observations and true wave:

    # Getting specified or random observation:
    if observation_number is None:
        row = np.random.randint(0, 100)  # Gets random observation row in advection_observation.csv file
        print('Getting random observation', row)
    else:
        row = observation_number    # Observation row in advection_observation.csv file
        print('Getting observation', row)

    # Observations:
    time_obs = pd.read_csv('advection_observations.csv', header=None, nrows=1).to_numpy()[0, 1:]  # Observation time
    x_obs = pd.read_csv('advection_observations.csv', header=None, nrows=1, skiprows=1).to_numpy()[0, 1:]  # Observation points
    NT_obs = time_obs.size  # Total number of observations (in time)
    NT_obs_steps = int(NT / NT_obs)  # Number of time steps until observation is made
    obs_dim = x_obs.size  # Observation dimensions
    Y = pd.read_csv('advection_observations.csv', header=None, skiprows=2).to_numpy()[row, 1:].reshape(obs_dim, NT_obs)  # Observations
    Y = Y.astype('float64')  # Change observation elements to float64 (for numpy computations)

    # True wave:
    time_true = pd.read_csv('advection_observations_n_true.csv', header=None, nrows=1).to_numpy()[0, 1:]  # Observation time
    x_true = pd.read_csv('advection_observations_n_true.csv', header=None, nrows=1, skiprows=1).to_numpy()[0, 1:]  # Observation points
    NT_true = time_true.size  # Total number of time steps for true wave
    true_dim = x_true.size  # Ttoal dimension size for true wave
    n_true = pd.read_csv('advection_observations_n_true.csv', header=None, skiprows=2).to_numpy()[row, 1:].reshape(150, NT)  # True wave
    n_true = n_true.astype('float64')  # Change true elements to float64 (for numpy computations)


    #######################################################
    # Constructing models:
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
    # Evolution model using Forward Euler's method:
    F = np.eye(N) + dt * R  # Using Forward Euler's method
    # Observation model:
    H = tools.compute_H(N, phi, x_obs, obs_dim)


    #######################################################
    # Constructing noise covariance for observation model:
    Gamma_v = np.zeros([NT, obs_dim, obs_dim])  # Initialising
    for k in range(NT):
        Gamma_v[k] = (sigma_v ** 2) * np.eye(obs_dim)  # Observation noise covariance

    #######################################################
    # Constructing noise covariances for evolution model:
    # For alpha:
    sigma_alpha_w = np.array([sigma_alpha_w_0, sigma_alpha_w_1, sigma_alpha_w_2])  # Array of standard deviations
    Gamma_alpha_w = basic_tools.compute_correlated_covariance_matrix(N, Np, Ne, sigma_alpha_w, sigma_alpha_w_correlation)  # Covariance matrix computation

    # Assimilation:
    Gamma_w = np.zeros([N, N])  # Initialising noise covariance for state
    Gamma_w[0:N, 0:N] = Gamma_alpha_w  # Adding alpha covariance to state covariance


    #######################################################
    # Computing prior:
    # For alpha:
    alpha_prior = tools.compute_coefficients(initial_condition, N, Np, phi, x_boundaries, h)  # Computing alpha coefficients from initial condition function
    Gamma_alpha_prior = np.eye(N)
    for i in range(N):
        if i % Np == 0:
            Gamma_alpha_prior[i, i] = (sigma_alpha_prior_0 ** 2)
        elif i % Np == 1:
            Gamma_alpha_prior[i, i] = (sigma_alpha_prior_1 ** 2)
        elif i % Np == 2:
            Gamma_alpha_prior[i, i] = (sigma_alpha_prior_2 ** 2)

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
    if use_BAE:
        BAE_data = np.load('advection_BAE.npz')  # Loading BAE data
        BAE_mean, BAE_covariance = BAE_data['BAE_mean'], BAE_data['BAE_covariance']  # Extracting mean and covariance from data
        model = Kalman_filter(F, H, Gamma_w, Gamma_v, NT, NT_obs=NT_obs, mean_epsilon=BAE_mean, Gamma_epsilon=BAE_covariance)
    else:
        model = Kalman_filter(F, H, Gamma_w, Gamma_v, NT, NT_obs=NT_obs)


    #######################################################
    # Computing time evolution of model:
    print('Computing Kalman filter estimates...')
    num = 0  # Initialising index for state
    t = np.zeros(NT)  # Initialising time array
    for j in range(NT_obs - 1):  # Iterating over total number of observation:
        for k in range(NT_obs_steps):  # Iterating over number of time steps between observations:
            num = j * NT_obs_steps + k  # Computing index for state evolution
            t[num + 1] = (num + 1) * dt  # Time (hours)
            x_predict[:, num + 1], Gamma_predict[num + 1] = model.predict(x[:, num], Gamma[num], num)  # Computing prediction
            x[:, num + 1], Gamma[num + 1] = x_predict[:, num + 1], Gamma_predict[num + 1]  # Adding prediction to state
        x[:, num + 1], Gamma[num + 1] = model.update(x[:, num + 1], Gamma[num + 1], Y[:, j + 1], j)  # Computing update
    # Predicting from last observation to end time T:
    for k in range(NT_obs_steps - 1):
        num = (NT_obs - 1) * NT_obs_steps + k  # Computing index for state
        t[num + 1] = (num + 1) * dt  # Time (hours)
        x_predict[:, num + 1], Gamma_predict[num + 1] = model.predict(x[:, num], Gamma[num], num)  # Computing prediction
        x[:, num + 1], Gamma[num + 1] = x_predict[:, num + 1], Gamma_predict[num + 1]  # Adding prediction to state


    #######################################################
    # Computing smoothed estimates:
    if smoothing:
        print('Computing Kalman smoother estimates...')
        x, Gamma = compute_fixed_interval_Kalman_smoother(F, NT, N, x, Gamma, x_predict, Gamma_predict)


    #######################################################
    # Extracting alpha and covariances from state:
    alpha = x[0:N, :]  # Wave coefficients
    Gamma_alpha = Gamma[:, 0:N, 0:N]  # Size distribution covariance


    #######################################################
    # Computing plotting discretisation:
    print('Computing plotting discretisation...')
    x_plot, n_plot, sigma_n = tools.get_plotting_discretisation_with_uncertainty(alpha, Gamma_alpha, xmin, xmax, phi, N, 150)
    n_plot_upper = n_plot + 2 * sigma_n
    n_plot_lower = n_plot - 2 * sigma_n


    #######################################################
    # Computing advection plotting discretisation:
    advection_plot = np.zeros([150, NT])  # Initialising
    for k in range(NT):  # Iterating over time
        for i in range(150):  # Iterating over space
            advection_plot[i, k] = advection(x_plot[i])

    #######################################################
    # Computing observation plotting discretisation:
    Y_plot = np.zeros([obs_dim, NT])  # Initialising
    for j in range(NT_obs):  # Iterating over total number of observation:
        for k in range(NT_obs_steps):  # Iterating over number of time steps between observations:
            num = j * NT_obs_steps + k  # Computing index for state
            Y_plot[:, num] = Y[:, j]  # Duplicating observations for plotting


    #######################################################
    # Computing norm difference between n_plot and n_true:
    n_diff = n_plot - n_true  # Computing (n - n_true)
    norm_diff = np.zeros(NT)  # Initialising norm difference for each time step
    if compute_weighted_norm:
        sigma_n = sigma_n + 1e-5  # Adding small number to sigma (to make non-singular)
        sigma_n[-1, :] = np.min(sigma_n[np.nonzero(sigma_n)])  # Adding smallest value to last row (to make non-singular)
        for k in range(NT):  # Iterating over time
            norm_diff[k] = np.sqrt(np.matmul(n_diff[:, k], np.matmul(np.diag(1 / (sigma_n[:, k] ** 2)), n_diff[:, k])))  # Computing norm
        print('Total weighted norm difference between estimate and truth:', str(round(np.linalg.norm(norm_diff), 4)))
    else:
        for k in range(NT):  # Iterating over time
            norm_diff[k] = np.sqrt(np.matmul(n_diff[:, k], n_diff[:, k]))  # Computing norm
        print('Total norm difference between estimate and truth:', str(round(np.linalg.norm(norm_diff), 4)))


    #######################################################
    # Printing total computation time:
    computation_time = round(tm.time() - initial_time, 3)  # Initial time stamp
    print('Total computation time:', str(computation_time), 'seconds.')  # Print statement


    #######################################################
    # Plotting animation and function parameters:

    # General parameters:
    location = 'Home'  # Set to 'Uni', 'Home', or 'Middle' (default)

    # Parameters for wave animation:
    xscale = 'linear'  # x-axis scaling ('linear' or 'log')
    xlimits = [xmin, xmax]  # Plot boundary limits for x-axis
    ylimits = [-0.2, 1.2]  # Plot boundary limits for y-axis
    xlabel = '$x$'  # x-axis label for 1D animation plot
    ylabel = '$n(x, t)$'  # y-axis label for 1D animation plot
    legend = ['Estimate', '$\pm 2 \sigma$', '', 'Truth', 'Simulated observations']  # Adding legend to plot
    legend_position = 'upper left'  # Position of legend in plot
    line_color = ['blue', 'blue', 'blue', 'green', 'red']  # Colors of lines in plot
    line_style = ['solid', 'dashed', 'dashed', 'solid', 'solid']  # Style of lines in plot
    time = t  # Array where time[i] is plotted (and animated)
    timetext = ('Time = ', '')  # Tuple where text to be animated is: timetext[0] + 'time[i]' + timetext[1]
    delay = 0  # Speed of animation (higher = slower)
    if use_BAE:
        title = 'Advection by DG method using BAE'  # Title for 1D animation plot
    else:
        title = 'Advection by DG method'  # Title for 1D animation plot

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
    basic_tools.plot_1D_animation(x_plot, n_plot, n_plot_lower, n_plot_upper, n_true, plot_add=(x_obs, Y_plot), xlimits=xlimits, ylimits=ylimits, xscale=xscale, xlabel=xlabel, ylabel=ylabel, title=title,
                                  location=location, legend=legend, legend_position=legend_position, time=time, timetext=timetext, line_color=line_color, line_style=line_style, doing_mainloop=False, delay=delay)

    # Advection animation:
    basic_tools.plot_1D_animation(x_plot, advection_plot, xlimits=xlimits_advection, ylimits=ylimits_advection, xscale=xscale_advection, xlabel=xlabel_advection, ylabel=ylabel_advection, title=title_advection,
                                  location=location_advection, line_color=line_color_advection, time=time_advection, timetext=timetext_advection, doing_mainloop=False, delay=delay_advection)

    # Mainloop:
    if plot_animations:
        print('Plotting...')
        mainloop()  # Runs tkinter GUI for plots and animations

    # Plotting norm difference:
    plt.figure(1)
    plt.plot(t, norm_diff)
    plt.xlim([0, T])
    plt.ylim([0, np.max(norm_diff)])
    plt.xlabel('$t$', fontsize=15)
    plt.ylabel(r'||$n_{est}(x, t) - n_{true}(x, t)$||$_W$', fontsize=14)
    plt.grid()
    plot_title = 'norm difference between truth and mean estimate'
    if compute_weighted_norm:
        plot_title = 'Weighted ' + plot_title
    if use_BAE:
        plot_title = plot_title + ' using BAE'
    plt.title(plot_title, fontsize=12)


    # Final prints:
    basic_tools.print_lines()  # Print lines in console
    print()  # Print space in console

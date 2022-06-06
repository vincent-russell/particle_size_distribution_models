"""

Title: Simulating observations of waves by the advection equation using the Discontinuous-Galerkin method
Author: Vincent Russell
Date: August 21, 2021

"""


#######################################################
# Modules:
import numpy as np
from numpy.random import normal
import time as tm
from tqdm import tqdm

# Local modules:
import basic_tools
import basic_models.advection_models.tools as tools


#######################################################
if __name__ == '__main__':

    #######################################################
    # Initial print:
    basic_tools.print_lines()


    #######################################################
    # Fixed parameters:

    # Spatial domain:
    xmin = 0  # Minimum
    xmax = 1  # Maximum

    # Time domain:
    dt = 0.001  # Time step
    T = 1  # End time
    NT = int(T / dt)  # Total number of time steps

    # Solution discretisation:
    Ne = 50  # Number of elements
    Np = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
    N = Ne * Np  # Total degrees of freedom

    # Observation parameters:
    obs_dim = 20  # Observation dimension size
    NT_obs_steps = 10  # Number of time steps until observation is made
    dt_obs = dt * NT_obs_steps  # Time step between observations
    NT_obs_total = int(T / dt_obs)  # Total number of observations (in time)

    # Loop parameters:
    number_iterations = 2  # Number of observations to compute


    #######################################################
    # Initialising timer for total computation:
    initial_time = tm.time()  # Initial time stamp


    #######################################################
    # Pre-loop computations:
    # Initialisations:
    alpha = np.zeros([N, NT])  # Initialising alpha = [alpha_0, alpha_1, ..., alpha_NT]
    # Discretisation:
    x_boundaries, h = tools.get_discretisation(Ne, xmin, xmax)
    # Basis functions:
    phi = tools.get_Legendre_basis(N, Np, x_boundaries)
    dphi = tools.get_Legendre_basis_derivative(N, Np, x_boundaries, phi)
    # Tensors:
    M = tools.compute_M(N, Np, h)


    #######################################################
    # Start of loop:
    print('Simulating', number_iterations, 'observations...')
    for iteration in tqdm(range(number_iterations)):

        #######################################################
        # Observation parameters:
        sigma_v = 0.2  # Covariance of noise = sigma_v^2 * I_M

        #######################################################
        # Initial condition parameters:
        N_0 = 1  # Amplitude of initial condition gaussian
        mu_0 = 0.2  # Mean of initial condition gaussian
        sigma_0 = 0.04  # Standard deviation of initial condition gaussian
        def initial_condition(x):
            return basic_tools.gaussian(x, N_0, mu_0, sigma_0)

        #######################################################
        # Advection parameters:
        c_0 = 0  # Constant coefficient
        c_1 = 3.5  # Linear coefficient
        c_2 = -3.5  # Quadratic coefficient
        def advection(x):
            return c_0 + c_1 * x + c_2 * x ** 2

        #######################################################
        # Evolution model computations:
        Q = tools.compute_Q(advection, N, Np, x_boundaries, phi, dphi)
        R = tools.compute_R(advection, Ne, Np, N, x_boundaries, phi, M, Q)
        # Evolution model using Crank-Nicolson method:
        F = np.matmul(np.linalg.inv(2 * np.eye(N) - dt * R), 2 * np.eye(N) + dt * R)

        #######################################################
        # Computing initial condition:
        alpha[:, 0] = tools.compute_coefficients(initial_condition, N, Np, phi, x_boundaries, h)  # Computing coefficients from initial condition function

        #######################################################
        # Computing time evolution of model:
        time = np.zeros(NT)  # Initialising time array
        for k in range(NT - 1):  # Iterating over time
            alpha[:, k + 1] = np.matmul(F, alpha[:, k])  # Computing next step
            time[k + 1] = k * dt  # Computing time

        #######################################################
        # Computing true size distribution and observation discretisation:
        x_true, n_true = tools.get_plotting_discretisation(alpha, xmin, xmax, NT, phi, Np, Ne, x_boundaries, h, 150)
        x_obs, n_obs = tools.get_plotting_discretisation(alpha, xmin, xmax, NT, phi, Np, Ne, x_boundaries, h, obs_dim)
        time_obs = np.zeros(NT_obs_total)  # Initialising observation time
        Y = np.zeros([obs_dim, NT_obs_total])  # Initialising observations
        for j in range(NT_obs_total):
            Y[:, j] = n_obs[:, j * NT_obs_steps] + normal(np.zeros(obs_dim), sigma_v)  # Computes y_k = n_k + v_k
            time_obs[j] = j * dt_obs  # Observation time

        #######################################################
        # Saving observations time and points in first row of .csv data:
        if iteration == 0:
            basic_tools.save_array_to_csv('advection_observations', time_obs, row_name='Observation time:')
            basic_tools.save_array_to_csv('advection_observations', x_obs, row_name='Observation points:')
            basic_tools.save_array_to_csv('advection_observations_n_true', time, row_name='Observation time:')
            basic_tools.save_array_to_csv('advection_observations_n_true', x_true, row_name='Observation points:')

        #######################################################
        # Saving observations and true size distribution to .csv data:
        observation_row_name = 'Observation ' + str(iteration) + ':'
        basic_tools.save_array_to_csv('advection_observations', Y, row_name=observation_row_name)
        basic_tools.save_array_to_csv('advection_observations_n_true', n_true, row_name=observation_row_name)


    #######################################################
    # Printing total computation time:
    computation_time = round(tm.time() - initial_time, 3)  # Initial time stamp
    print('Total computation time:', str(computation_time), 'seconds.')  # Print statement


    #######################################################
    # Final prints:
    basic_tools.print_lines()  # Print lines in console
    print()  # Print space in console

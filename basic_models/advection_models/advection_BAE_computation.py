"""

Title: Computing the Bayesian approximation error for the state estimation/identification in the advection equation
Author: Vincent Russell
Date: August 26, 2021

"""


#######################################################
# Modules:
import random

import numpy as np
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
    # Solution discretisation for reduced model:
    Ne_r = 10  # Number of elements (needs to be a multiple of Ne)
    Np_r = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
    N_r = Ne_r * Np_r  # Total degrees of freedom

    # Loop parameters:
    N_iterations = 1000  # Number of samples from prior to compute BAE


    #######################################################
    # Initialising timer for total computation:
    initial_time = tm.time()  # Initial time stamp


    #######################################################
    # Pre-loop computations:
    # Initialisations:
    epsilon = np.zeros([N_iterations, N_r, NT])  # Initialising BAE
    alpha = np.zeros([N, NT])  # Initialising alpha = [alpha_0, alpha_1, ..., alpha_NT]
    # Discretisation:
    x_boundaries, h = tools.get_discretisation(Ne, xmin, xmax)
    x_boundaries_r, h_r = tools.get_discretisation(Ne_r, xmin, xmax)
    # Basis functions:
    phi = tools.get_Legendre_basis(N, Np, x_boundaries)
    dphi = tools.get_Legendre_basis_derivative(N, Np, x_boundaries, phi)
    phi_r = tools.get_Legendre_basis(N_r, Np_r, x_boundaries_r)
    dphi_r = tools.get_Legendre_basis_derivative(N_r, Np_r, x_boundaries_r, phi_r)
    # Tensors:
    M = tools.compute_M(N, Np, h)
    M_r = tools.compute_M(N_r, Np_r, h_r)
    # Projection operator:
    G = tools.compute_G(N, N_r, Np, Np_r, phi, phi_r, x_boundaries)
    P = np.matmul(np.linalg.inv(M_r), G)


    #######################################################
    # Start of loop:
    print('Computing', N_iterations, 'simulations for BAE...')
    for iteration in tqdm(range(N_iterations)):

        #######################################################
        # Initial condition parameters:
        N_0 = 1  # Amplitude of initial condition gaussian
        mu_0 = 0.2  # Mean of initial condition gaussian
        sigma_0 = 0.04  # Standard deviation of initial condition gaussian
        def initial_condition(x):
            return basic_tools.gaussian(x, N_0, mu_0, sigma_0)

        #######################################################
        # Advection parameters:

        # Advection model:
        c_0 = 0  # Constant coefficient
        c_1 = random.uniform(2.8, 4.2)  # Linear coefficient
        c_2 = -c_1  # Quadratic coefficient
        def advection(x):
            return c_0 + c_1 * x + c_2 * x ** 2

        # Reduced advection model:
        c_0_r = 0.4  # Constant coefficient
        c_1_r = 0  # Linear coefficient
        c_2_r = 0  # Quadratic coefficient
        def advection_r(x):
            return c_0_r + c_1_r * x + c_2_r * x ** 2

        #######################################################
        # Evolution model computations:
        Q = tools.compute_Q(advection, N, Np, x_boundaries, phi, dphi)
        Q_r = tools.compute_Q(advection_r, N_r, Np_r, x_boundaries_r, phi_r, dphi_r)
        R = tools.compute_R(advection, Ne, Np, N, x_boundaries, phi, M, Q)
        R_r = tools.compute_R(advection_r, Ne_r, Np_r, N_r, x_boundaries_r, phi_r, M_r, Q_r)
        # Evolution model:
        F = np.eye(N) + dt * R
        # Reduced evolution model:
        F_r = np.eye(N_r) + dt * R_r

        #######################################################
        # BAE computation function:
        compute_epsilon_matrix = np.matmul(P, F) - np.matmul(F_r, P)

        #######################################################
        # Computing initial condition:
        alpha[:, 0] = tools.compute_coefficients(initial_condition, N, Np, phi, x_boundaries, h)  # Computing coefficients from initial condition function
        epsilon[iteration, :, 0] = np.matmul(compute_epsilon_matrix, alpha[:, 0])  # Computing initial epsilon

        #######################################################
        # Computing BAE for each time step:
        for k in range(NT - 1):  # Iterating over time
            alpha[:, k + 1] = np.matmul(F, alpha[:, k])
            epsilon[iteration, :, k + 1] = np.matmul(compute_epsilon_matrix, alpha[:, k + 1])


    #######################################################
    # Computing BAE sample mean and sample covariance over all iterations:
    print('Computing BAE sample mean and covariance...')
    BAE_mean = np.average(epsilon, axis=0)  # Computing BAE mean
    BAE_covariance = np.zeros([NT, N_r, N_r])  # Initialising
    for k in range(NT):  # Iterating over time
        epsilon_difference = np.zeros([N_r, 1])  # Initialising difference vector
        epsilon_difference_matrix = np.zeros([N_iterations, N_r, N_r])  # Initialising difference matrix
        for iteration in range(N_iterations):
            epsilon_difference[:, 0] = epsilon[iteration, :, k] - BAE_mean[:, k]  # Computing epsilon(i)_k - mu_epsilon_k
            epsilon_difference_matrix[iteration] = np.matmul(epsilon_difference, np.transpose(epsilon_difference))  # Computing (epsilon(i)_k - mu_epsilon_k) * (epsilon(i)_k - mu_epsilon_k)^T
        BAE_covariance[k] = (1 / (N_iterations - 1)) * np.sum(epsilon_difference_matrix, axis=0)  # Computing BAE covariance


    #######################################################
    # Saving BAE mean and covariance:
    print('Saving BAE sample mean and covariance...')
    np.savez('advection_BAE', BAE_mean=BAE_mean, BAE_covariance=BAE_covariance)  # Saving BAE data in .npz file


    #######################################################
    # Printing total computation time:
    computation_time = round(tm.time() - initial_time, 3)  # Initial time stamp
    print('Total computation time:', str(computation_time), 'seconds.')  # Print statement


    #######################################################
    # Final prints:
    basic_tools.print_lines()  # Print lines in console
    print()  # Print space in console

"""

Title: Estimating VAR(2) coefficients from simulated data (in R^N)
Author: Vincent Russell
Date: August 10, 2021

Note: These codes are tailored to the log-spaced GDE state identification problem

"""


#######################################################
# Modules:
import numpy as np
import time as tm
import matplotlib.pyplot as plt
from tqdm import tqdm

# Local modules:
import basic_tools
from basic_tools import gaussian
from evolution_models.tools import (get_discretisation, get_Legendre_basis, log_transform_function_check,
                                    compute_coefficients)


#######################################################
if __name__ == '__main__':

    #######################################################
    # Parameters:

    # Setup parameters:
    scale_type = 'linear'
    plotting_series = False
    save_VAR_coefficients = True
    save_name = 'eta_2_coefficients'

    # Spatial domain:
    Dp_min = 1  # Minimum diameter of particles (micro m)
    Dp_max = 11  # Maximum diameter of particles (micro m)
    vmin = basic_tools.diameter_to_volume(Dp_min)  # Minimum volume of particles (micro m^3)
    vmax = basic_tools.diameter_to_volume(Dp_max)  # Maximum volume of particles (micro m^3)

    # Discretisation:
    Ne = 5  # Number of elements
    Np = 3  # Np - 1 = degree of Legendre polynomial approximation in each element
    N = Ne * Np  # Total degrees of freedom

    # Time domain:
    dt = (1 / 60) * 20  # Time step (hours)
    T = 24  # End time (hours)
    NT = int(T / dt)  # Total number of time steps

    # Autoregressive model parameters:
    VAR_p = 2  # Order of VAR(p) model

    # Noise model (numbers are for order, i.e. e_list should be len(Np)):
    e_0 = 0.15 / 0.1
    e_1 = e_0 / 2
    e_2 = e_1 / 4
    e_list = [e_0, e_1, e_2]

    # Condensation Function to approximate by a VAR(p) model:
    # I_0 = 0.2  # Condensation parameter constant
    # I_1 = 1  # Condensation parameter inverse quadratic
    # def function_time(_):
    #     def function(Dp):
    #         return I_0 + I_1 / (Dp ** 2)
    #     return function

    # Deposition function to approximate by a VAR(p) model:
    depo_Dpmin = 5  # Deposition parameter; diameter at which minimum
    d_0 = 0.4  # Deposition parameter constant
    d_1 = -0.15  # Deposition parameter linear
    d_2 = -d_1 / (2 * depo_Dpmin)  # Deposition parameter quadratic
    def function_time(_):
        def function(Dp):
            return d_0 + d_1 * Dp + d_2 * Dp ** 2  # Quadratic model output
        return function


    #######################################################
    # Initialising timer for total computation:
    print(), basic_tools.print_lines(), print('Initialising...')
    initial_time = tm.time()  # Initial time stamp


    #######################################################
    # Computing noise vector:
    e_vector = np.zeros(N)  # Initialising
    for ell in range(Ne):
        for j in range(Np):
            e_vector[ell * Np + j] = e_list[j]


    #######################################################
    # Computing discretisation:
    _, x_boundaries, _ = get_discretisation(Ne, Np, Dp_min, Dp_max)
    h = np.zeros(Ne)
    for ell in range(Ne):
        h[ell] = x_boundaries[ell + 1] - x_boundaries[ell]
    phi = get_Legendre_basis(N, Np, x_boundaries)


    #######################################################
    # Computing time series from function values:
    print('Simulating time series from function...')
    x_simulated = np.zeros([N, NT])  # Initialising time series
    t = np.zeros(NT)  # Initialising time array
    for k in tqdm(range(NT - 1)):
        g = log_transform_function_check(function_time(t[k]), scale_type)
        x_simulated[:, k] = compute_coefficients(g, N, Np, phi, x_boundaries, h) + np.random.normal(scale=e_vector)
        t[k + 1] = (k + 1) * dt  # Time (hours)


    #######################################################
    # Constructing matrix Phi such that y = Phi * x, where x = [a1_11, a1_12, a1_21, a1_22, a2_11, a2_12, a2_21, a2_22]:
    print('Constructing matrix Phi...')
    Phi = np.zeros([N * (NT - VAR_p), (N * N) * VAR_p])  # Initialising
    for k in range(NT - VAR_p):
        for p in range(VAR_p):
            index = (k + (VAR_p - 1)) - p
            for i in range(N):
                Phi[N * k + i, i * N + N * N * p: i * N + N * N * p + N] = x_simulated[:, index]


    #######################################################
    # Constructing vector y such that y = Phi * x:
    print('Constructing vector y...')
    y = np.zeros([N * (NT - VAR_p)])
    for k in range(NT - VAR_p):
        y[k * N: (k + 1) * N] = x_simulated[:, k + VAR_p]


    #######################################################
    # Estimating A_coef = min_x ||y = Phi * x||, where where x = [a1_11, a1_12, a1_21, a1_22, a2_11, a2_12, a2_21, a2_22]:
    print('Estimating A1 and A2 from simulated time series...')
    Phi_T = np.transpose(Phi)
    inv_Phi_T_Phi = np.linalg.inv(np.matmul(Phi_T, Phi))
    Phi_T_y = np.matmul(Phi_T, y)
    A_estimate = np.matmul(inv_Phi_T_Phi, Phi_T_y)
    A_estimate_array = np.zeros([VAR_p, N, N])
    for p in range(VAR_p):
        A_estimate_array[p] = A_estimate.reshape(VAR_p, N, N)[p]


    #######################################################
    # Printing total computation time:
    computation_time = round(tm.time() - initial_time, 3)  # Initial time stamp
    print('Total computation time:', str(computation_time), 'seconds.')  # Print statement
    basic_tools.print_lines()


    #######################################################
    # Printing results:
    for p in range(VAR_p):
        print('A' + str(p + 1) + ' estimate:', np.round(A_estimate_array[p].reshape(1, N * N), 2))


    #######################################################
    # Saving VAR(p) coefficients:
    if save_VAR_coefficients:
        print('Saving coefficients...')
        np.savez(save_name, A_estimate_array=A_estimate_array)
    basic_tools.print_lines()


    #######################################################
    # Plotting simulated time series:

    if plotting_series:
        print('Plotting...')
        for i in range(N):
            plt.figure(i)
            plt.plot(x_simulated[i, :])
            plt.title('Element ' + str(i), fontsize=14)
            plt.xlim([0, NT])
            plt.ylim([np.min(x_simulated[i, :]), np.max(x_simulated[i, :])])
            plt.ylabel('Element ' + str(i), fontsize=14)
            plt.xlabel('$t$', fontsize=14)
            plt.grid()

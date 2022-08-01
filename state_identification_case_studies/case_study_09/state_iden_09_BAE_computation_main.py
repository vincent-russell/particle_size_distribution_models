"""

Title: BAE computation for the state identification of aerosol particle size distribution
Author: Vincent Russell
Date: June 27, 2022

"""


#######################################################
# Modules:
import random
import numpy as np
import time as tm
from tqdm import tqdm

# Local modules:
import basic_tools
from evolution_models.tools import GDE_evolution_model, GDE_Jacobian, compute_G


#######################################################
# Importing parameter file:
from state_identification_case_studies.case_study_09.state_iden_09_BAE_computation_parameters import *


#######################################################
if __name__ == '__main__':

    #######################################################
    # Initialising timer for total computation:
    initial_time = tm.time()  # Initial time stamp


    #######################################################
    # Print statement:
    basic_tools.print_lines()
    print('Initialising computation for BAE...')


    #######################################################
    # Initialising evolution model and reduced evolution model:
    F_alpha = GDE_evolution_model(Ne, Np, xmin, xmax, dt, NT, boundary_zero=boundary_zero, scale_type='log', print_status=False)  # Initialising evolution model
    F_alpha_r = GDE_evolution_model(Ne_r, Np_r, xmin, xmax, dt, NT, boundary_zero=boundary_zero, scale_type='log', print_status=False)  # Initialising reduced evolution model


    #######################################################
    # Continuity constraint computations for gamma (condensation):
    num_constraints_gamma = Ne_gamma - 1  # Number of contraints
    Nc_gamma = N_gamma - num_constraints_gamma  # Dimensions of constrained gamma


    #######################################################
    # Initialising approximation error, and computing projection matrix:

    # Initialising approximation error:
    epsilon = np.zeros([N_iterations, N_r + gamma_p * Nc_gamma + J_p, NT])  # Initialising BAE

    # For alpha:
    G_alpha = compute_G(N, N_r, Np, Np_r, F_alpha.phi, F_alpha_r.phi, F_alpha.x_boundaries)  # Computing matrix G for projection operator
    P_alpha = np.matmul(np.linalg.inv(F_alpha_r.M), G_alpha)  # Computing projection operator

    # Assimilation:
    P = np.zeros([N_r + gamma_p * Nc_gamma + J_p, N])  # Initialising projection operator
    P[0:N_r, 0:N] = P_alpha  # Adding alpha projection


    #######################################################
    # Start of loop for samples of epsilon:
    print('Computing', N_iterations, 'simulations for BAE...')
    for iteration in tqdm(range(N_iterations)):

        #######################################################
        # Drawing sample of parameters for evolution model:

        # Sample of the initial condensation rate I_0(Dp) = I_Dp(Dp, 0):
        I_cst = random.uniform(0.001, 0.003)  # Condensation parameter constant
        I_linear = random.uniform(0, 0.1)  # Condensation parameter linear
        def cond(Dp):
            return I_cst + I_linear * Dp

        # Sample of the deposition rate d(Dp):
        d_cst = random.uniform(0, 0.1)  # Deposition parameter constant
        d_linear = random.uniform(0, 0.1)  # Deposition parameter linear
        d_inverse_quadratic = random.uniform(0, 0.000003)  # Deposition parameter inverse quadratic
        def depo(Dp):
            return d_cst + d_linear * Dp + d_inverse_quadratic * (1 / Dp ** 2)

        # Sample of the source (nucleation event) model:
        N_s = random.uniform(1.2e3, 1.8e3)  # Amplitude of gaussian nucleation event
        t_s = random.uniform(7, 9)  # Mean time of gaussian nucleation event
        sigma_s = random.uniform(1.3, 1.7)  # Standard deviation time of gaussian nucleation event
        def sorc(t):  # Source (nucleation) at xmin
            return basic_tools.gaussian(t, N_s, t_s, sigma_s)  # Gaussian source (nucleation event) model output

        #######################################################
        # Constructing size distribution evolution model:
        F_alpha.add_process('condensation', cond)  # Adding condensation to evolution model
        F_alpha.add_process('deposition', depo)  # Adding deposition to evolution model
        F_alpha.add_process('source', sorc)  # Adding source to evolution model
        F_alpha.add_process('coagulation', coag, load_coagulation=load_coagulation, coagulation_suffix=coagulation_suffix)  # Adding coagulation to evolution model
        F_alpha.compile()  # Compiling evolution model

        #######################################################
        # Constructing reduced size distribution evolution model:
        F_alpha_r.add_process('condensation', cond)  # Adding condensation to evolution model
        F_alpha_r.add_process('deposition', guess_depo)  # Adding deposition to evolution model
        F_alpha_r.add_process('source', sorc)  # Adding source to evolution model
        F_alpha_r.compile()  # Compiling evolution model

        #######################################################
        # Constructing Jacobian of size distribution evolution model:
        J_F_alpha = GDE_Jacobian(F_alpha)  # Evolution Jacobian
        J_F_alpha_r = GDE_Jacobian(F_alpha_r)  # Evolution Jacobian
        dF_alpha_d_alpha = J_F_alpha.eval_d_alpha  # Derivative with respect to alpha
        dF_alpha_d_alpha_r = J_F_alpha_r.eval_d_alpha  # Derivative with respect to alpha

        #######################################################
        # Functions to compute/update reduced evolution operator using Crank-Nicolson method:

        # Function to compute evolution operator:
        def compute_evolution_operator(alpha_star, t_star):
            # Computing J_star:
            J_alpha_star = dF_alpha_d_alpha(alpha_star, t_star)
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

        # Function to compute evolution operator:
        def compute_evolution_operator_r(x_star_r, t_star):
            # Extracting alpha:
            alpha_star_r = x_star_r[0:N_r]
            # COmputing J_star:
            J_alpha_star_r = dF_alpha_d_alpha_r(alpha_star_r, t_star)
            # Computing F_star:
            F_star_r = F_alpha_r.eval(alpha_star_r, t_star) - np.matmul(J_alpha_star_r, alpha_star_r)
            # Computing evolution operators for each coefficient:
            matrix_multiplier = np.linalg.inv(np.eye(N_r) - (dt / 2) * J_alpha_star_r)  # Computing matrix multiplier for evolution operators and additive vector
            F_evol_alpha_r = np.matmul(matrix_multiplier, (np.eye(N_r) + (dt / 2) * J_alpha_star_r))
            # Computing evolution operator:
            F_evolution_r = np.zeros([N_r + gamma_p * Nc_gamma + J_p, N_r + gamma_p * Nc_gamma + J_p])  # Initialising
            F_evolution_r[0:N_r, 0:N_r] = F_evol_alpha_r
            # Computing evolution additive vector:
            b_evolution_r = np.zeros(N_r + gamma_p * Nc_gamma + J_p)  # Initialising
            b_evolution_r[0:N_r] = np.matmul(matrix_multiplier, (dt * F_star_r))
            return F_evolution_r, b_evolution_r

        #######################################################
        # Initialising evolution operator and additive evolution vector:
        F = np.zeros([NT, N, N])  # Initialising evolution operator F_0, F_1, ..., F_{NT - 1}
        F_r = np.zeros([NT, N_r + gamma_p * Nc_gamma + J_p, N_r + gamma_p * Nc_gamma + J_p])  # Initialising evolution operator F_0, F_1, ..., F_{NT - 1}
        b = np.zeros([N, NT])  # Initialising additive evolution vector b_0, b_1, ..., b_{NT - 1}
        b_r = np.zeros([N_r + gamma_p * Nc_gamma + J_p, NT])  # Initialising additive evolution vector b_0, b_1, ..., b_{NT - 1}

        #######################################################
        # Initialising states and adding priors:
        alpha = np.zeros([N, NT])  # Initialising
        x_r = np.zeros([N_r + gamma_p * Nc_gamma + J_p, NT])  # Initialising
        alpha[:, 0] = F_alpha.compute_coefficients('alpha', initial_guess_size_distribution)  # Computing alpha coefficients from initial condition function
        x_r[0:N_r, 0] = F_alpha_r.compute_coefficients('alpha', initial_guess_size_distribution)  # Computing alpha coefficients from initial condition function

        #######################################################
        # Computing time evolution and approximation error for each time step:
        t = np.zeros(NT)  # Initialising time array
        for k in range(NT - 1):  # Iterating over time
            F[k], b[:, k] = compute_evolution_operator(alpha[:, k], t[k])  # Computing evolution operator F and vector b
            F_r[k], b_r[:, k] = compute_evolution_operator_r(x_r[:, k], t[k])  # Computing evolution operator F and vector b
            compute_epsilon_matrix = np.matmul(P, F[k]) - np.matmul(F_r[k], P)  # Approximation error computation matrix
            compute_epsilon_vector = np.matmul(P, b[:, k]) - b_r[:, k]  # Approximation error computation vector
            epsilon[iteration, :, k] = np.matmul(compute_epsilon_matrix, alpha[:, k]) + compute_epsilon_vector  # Computing approximation error
            alpha[:, k + 1] = np.matmul(F[k], alpha[:, k]) + b[:, k]  # Time evolution computation
            x_r[:, k + 1] = np.matmul(F_r[k], x_r[:, k]) + b_r[:, k]  # Time evolution computation
            t[k + 1] = (k + 1) * dt  # Time (hours)
        # Computing approximation error at final time:
        compute_epsilon_matrix = np.matmul(P, F[NT - 1]) - np.matmul(F_r[NT - 1], P)  # Approximation error computation matrix
        compute_epsilon_vector = np.matmul(P, b[:, NT - 1]) - b_r[:, NT - 1]  # Approximation error computation vector
        epsilon[iteration, :, NT - 1] = np.matmul(compute_epsilon_matrix, alpha[:, NT - 1]) + compute_epsilon_vector  # Computing approximation error


    #######################################################
    # Computing BAE sample mean and sample covariance over all iterations:
    print('Computing BAE sample mean and covariance...')
    BAE_mean = np.average(epsilon, axis=0)  # Computing BAE mean
    BAE_covariance = np.zeros([NT, N_r + gamma_p * Nc_gamma + J_p, N_r + gamma_p * Nc_gamma + J_p])  # Initialising
    for k in range(NT):  # Iterating over time
        epsilon_difference = np.zeros([N_r + gamma_p * Nc_gamma + J_p, 1])  # Initialising difference vector
        epsilon_difference_matrix = np.zeros([N_iterations, N_r + gamma_p * Nc_gamma + J_p, N_r + gamma_p * Nc_gamma + J_p])  # Initialising difference matrix
        for iteration in range(N_iterations):
            epsilon_difference[:, 0] = epsilon[iteration, :, k] - BAE_mean[:, k]  # Computing epsilon(i)_k - mu_epsilon_k
            epsilon_difference_matrix[iteration] = np.matmul(epsilon_difference, np.transpose(epsilon_difference))  # Computing (epsilon(i)_k - mu_epsilon_k) * (epsilon(i)_k - mu_epsilon_k)^T
        BAE_covariance[k] = (1 / (N_iterations - 1)) * np.sum(epsilon_difference_matrix, axis=0)  # Computing BAE covariance


    #######################################################
    # Saving BAE mean and covariance:
    print('Saving BAE sample mean and covariance...')
    np.savez(filename_BAE, BAE_mean=BAE_mean, BAE_covariance=BAE_covariance)  # Saving BAE data in .npz file


    #######################################################
    # Printing total computation time:
    computation_time = round(tm.time() - initial_time, 3)  # Initial time stamp
    print('Total computation time:', str(computation_time), 'seconds.')  # Print statement


    #######################################################
    # Final prints:
    basic_tools.print_lines()  # Print lines in console
    print()  # Print space in console

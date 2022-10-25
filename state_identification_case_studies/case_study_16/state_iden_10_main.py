"""

Title: State identification of aerosol particle size distribution
Author: Vincent Russell
Date: June 27, 2022

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
from basic_tools import Kalman_filter, compute_fixed_interval_Kalman_smoother
from observation_models.data.simulated import load_observations
from evolution_models.tools import GDE_evolution_model, GDE_Jacobian, compute_U, change_basis_x_to_logDp, change_basis_x_to_logDp_sorc
from observation_models.tools import get_DMA_transfer_function, compute_alpha_to_z_operator, Size_distribution_observation_model


#######################################################
# Importing parameter file:
from state_identification_case_studies.case_study_10.state_iden_10_parameters import *


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
    # Constructing evolution model:
    F_alpha = GDE_evolution_model(Ne, Np, xmin, xmax, dt, NT, boundary_zero=boundary_zero, scale_type='log', discretise_with_diameter=discretise_with_diameter)  # Initialising evolution model
    F_alpha.add_unknown('condensation', Ne_gamma, Np_gamma)  # Adding condensation as unknown to evolution model
    F_alpha.add_unknown('deposition', Ne_eta, Np_eta)  # Adding deposition as unknown to evolution model
    F_alpha.add_unknown('source')  # Adding source as unknown to evolution model
    F_alpha.add_process('coagulation', coag, load_coagulation=load_coagulation, coagulation_suffix=coagulation_suffix)  # Adding coagulation to evolution model
    F_alpha.compile()  # Compiling evolution model


    #######################################################
    # Continuity constraint computations for gamma (condensation):
    num_constraints_gamma = Ne_gamma - 1  # Number of contraints
    Nc_gamma = N_gamma - num_constraints_gamma  # Dimensions of constrained gamma
    U_gamma, UT_gamma = compute_U(N_gamma, Ne_gamma, Np_gamma, F_alpha.phi_gamma, F_alpha.x_boundaries_gamma)  # Computing null space continuity matrix
    # Continuity constraint matrix with multiple states in time (accounting for VAR(p) model):
    U_gamma_p = np.zeros([gamma_p * N_gamma, gamma_p * Nc_gamma])  # Initialising
    UT_gamma_p = np.zeros([gamma_p * Nc_gamma, gamma_p * N_gamma])  # Initialising
    for i in range(gamma_p):
        U_gamma_p[i * N_gamma: (i + 1) * N_gamma, i * Nc_gamma: (i + 1) * Nc_gamma] = U_gamma
        UT_gamma_p[i * Nc_gamma: (i + 1) * Nc_gamma, i * N_gamma: (i + 1) * N_gamma] = UT_gamma


    #######################################################
    # Continuity constraint computations for eta (deposition):
    num_constraints_eta = Ne_eta - 1  # Number of contraints
    Nc_eta = N_eta - num_constraints_eta  # Dimensions of constrained eta
    U_eta, UT_eta = compute_U(N_eta, Ne_eta, Np_eta, F_alpha.phi_eta, F_alpha.x_boundaries_eta)  # Computing null space continuity matrix
    # Continuity constraint matrix with multiple states in time (accounting for VAR(p) model):
    U_eta_p = np.zeros([eta_p * N_eta, eta_p * Nc_eta])  # Initialising
    UT_eta_p = np.zeros([eta_p * Nc_eta, eta_p * N_eta])  # Initialising
    for i in range(eta_p):
        U_eta_p[i * N_eta: (i + 1) * N_eta, i * Nc_eta: (i + 1) * Nc_eta] = U_eta
        UT_eta_p[i * Nc_eta: (i + 1) * Nc_eta, i * N_eta: (i + 1) * N_eta] = UT_eta


    #######################################################
    # Constructing condensation rate VAR(p) evolution model:

    # A matrix (evolution operator for gamma_tilde):
    A_gamma = np.zeros([gamma_p * N_gamma, gamma_p * N_gamma])  # Initialising matrix
    for i in range(gamma_p - 1):
        A_gamma[0: N_gamma, (i * N_gamma): ((i + 1) * N_gamma)] = gamma_A[i]  # Computing elements
        A_gamma[((i + 1) * N_gamma): ((i + 2) * N_gamma), (i * N_gamma): ((i + 1) * N_gamma)] = np.eye(N_gamma)  # Adding identity to off-diagonal
    A_gamma[0: N_gamma, ((gamma_p - 1) * N_gamma): (gamma_p * N_gamma)] = gamma_A[gamma_p - 1]  # Computing last element

    # B matrix (modification for evolution operator for alpha):
    B_gamma = np.zeros([N_gamma, gamma_p * N_gamma])  # Initilising matrix
    for i in range(gamma_p):
        B_gamma[0: N_gamma, (i * N_gamma): ((i + 1) * N_gamma)] = gamma_A[i]  # Computing elements
    B_gamma[0: N_gamma, 0: N_gamma] = B_gamma[0: N_gamma, 0: N_gamma] + np.eye(N_gamma)  # Adding one to first element

    # C matrix (extracting current state from vector of multiple states):
    C_gamma = np.zeros([Nc_gamma, gamma_p * Nc_gamma])
    C_gamma[0: Nc_gamma, 0: Nc_gamma] = np.eye(Nc_gamma)


    #######################################################
    # Constructing deposition rate VAR(p) evolution model:

    # A matrix (evolution operator for eta_tilde):
    A_eta = np.zeros([eta_p * N_eta, eta_p * N_eta])  # Initialising matrix
    for i in range(eta_p - 1):
        A_eta[0: N_eta, (i * N_eta): ((i + 1) * N_eta)] = eta_A[i]  # Computing elements
        A_eta[((i + 1) * N_eta): ((i + 2) * N_eta), (i * N_eta): ((i + 1) * N_eta)] = np.eye(N_eta)  # Adding identity to off-diagonal
    A_eta[0: N_eta, ((eta_p - 1) * N_eta): (eta_p * N_eta)] = eta_A[eta_p - 1]  # Computing last element

    # B matrix (modification for evolution operator for alpha):
    B_eta = np.zeros([N_eta, eta_p * N_eta])  # Initilising matrix
    for i in range(eta_p):
        B_eta[0: N_eta, (i * N_eta): ((i + 1) * N_eta)] = eta_A[i]  # Computing elements
    B_eta[0: N_eta, 0: N_eta] = B_eta[0: N_eta, 0: N_eta] + np.eye(N_eta)  # Adding one to first element

    # C matrix (extracting current state from vector of multiple states):
    C_eta = np.zeros([Nc_eta, eta_p * Nc_eta])
    C_eta[0: Nc_eta, 0: Nc_eta] = np.eye(Nc_eta)


    #######################################################
    # Constructing source (nucleation) rate AR(p) evolution model:

    # A matrix:
    A_J = np.zeros([J_p, J_p])  # Initialising matrix
    for i in range(J_p):
        A_J[0, i] = J_a[i]  # Computing elements
    A_J = A_J + np.eye(J_p, J_p, -1)  # Adding ones to off-diagonal

    # B matrix:
    B_J = np.zeros([1, J_p])  # Initilising matrix
    for i in range(J_p):
        B_J[0, i] = J_a[i]  # Computing elements
    B_J[0, 0] = B_J[0, 0] + 1  # Adding one to first element


    #######################################################
    # Constructing Jacobian of size distribution evolution model:
    J_F_alpha = GDE_Jacobian(F_alpha)  # Evolution Jacobian
    dF_alpha_d_alpha = J_F_alpha.eval_d_alpha  # Derivative with respect to alpha
    dF_alpha_d_gamma = J_F_alpha.eval_d_gamma  # Derivative with respect to gamma
    dF_alpha_d_eta = J_F_alpha.eval_d_eta  # Derivative with respect to eta
    dF_alpha_d_J = J_F_alpha.eval_d_J  # Derivative with respect to J


    #######################################################
    # Functions to compute/update evolution operator, Jacobians, and covariance using Crank-Nicolson method:

    # Function to compute evolution operator Jacobians:
    def compute_evolution_operator_Jacobians(x_tilde_c_star, t_star):
        # Getting current state (i.e. accounting for VAR(p) model):
        x_c_star = np.zeros(N + Nc_gamma + Nc_eta + J_p)
        x_c_star[0: N] = x_tilde_c_star[0: N]
        x_c_star[N: N + Nc_gamma] = np.matmul(C_gamma, x_tilde_c_star[N: N + gamma_p * Nc_gamma])
        x_c_star[N + Nc_gamma: N + Nc_gamma + Nc_eta] = np.matmul(C_eta, x_tilde_c_star[N + gamma_p * Nc_gamma: N + gamma_p * Nc_gamma + eta_p * Nc_eta])
        x_c_star[N + Nc_gamma + Nc_eta: N + Nc_gamma + Nc_eta + J_p] = x_tilde_c_star[N + gamma_p * Nc_gamma + eta_p * Nc_eta: N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p]
        # Computing non-constrained state:
        x_star = np.zeros(N + N_gamma + N_eta + J_p)  # Initialising
        x_star[0: N] = x_c_star[0: N]  # Extracting alpha from constrained state x
        x_star[N: N + N_gamma] = np.matmul(U_gamma, x_c_star[N: N + Nc_gamma])  # Extracting gamma from constrained state x
        x_star[N + N_gamma: N + N_gamma + N_eta] = np.matmul(U_eta, x_c_star[N + Nc_gamma: N + Nc_gamma + Nc_eta])  # Extracting eta from constrained state x
        x_star[N + N_gamma + N_eta: N + N_gamma + N_eta + J_p] = x_c_star[N + Nc_gamma + Nc_eta: N + Nc_gamma + Nc_eta + J_p]  # Extracting J from constrained state x
        # Computing Jacobians:
        J_alpha_star = dF_alpha_d_alpha(x_star, t_star)
        J_gamma_star = dF_alpha_d_gamma(x_star, t_star)
        J_eta_star = dF_alpha_d_eta(x_star, t_star)
        J_J_star = dF_alpha_d_J(x_star, t_star)
        return J_alpha_star, J_gamma_star, J_eta_star, J_J_star

    # Function to compute evolution operator:
    def compute_evolution_operator(x_tilde_c_star, t_star, J_alpha_star, J_gamma_star, J_eta_star, J_J_star):
        # Getting current state (i.e. accounting for VAR(p) model):
        x_c_star = np.zeros(N + Nc_gamma + Nc_eta + J_p)
        x_c_star[0: N] = x_tilde_c_star[0: N]
        x_c_star[N: N + Nc_gamma] = np.matmul(C_gamma, x_tilde_c_star[N: N + gamma_p * Nc_gamma])
        x_c_star[N + Nc_gamma: N + Nc_gamma + Nc_eta] = np.matmul(C_eta, x_tilde_c_star[N + gamma_p * Nc_gamma: N + gamma_p * Nc_gamma + eta_p * Nc_eta])
        x_c_star[N + Nc_gamma + Nc_eta: N + Nc_gamma + Nc_eta + J_p] = x_tilde_c_star[N + gamma_p * Nc_gamma + eta_p * Nc_eta: N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p]
        # Computing non-constrained state:
        x_star = np.zeros(N + N_gamma + N_eta + J_p)  # Initialising
        x_star[0: N] = x_c_star[0: N]  # Extracting alpha from constrained state x
        x_star[N: N + N_gamma] = np.matmul(U_gamma, x_c_star[N: N + Nc_gamma])  # Extracting gamma from constrained state x
        x_star[N + N_gamma: N + N_gamma + N_eta] = np.matmul(U_eta, x_c_star[N + Nc_gamma: N + Nc_gamma + Nc_eta])  # Extracting eta from constrained state x
        x_star[N + N_gamma + N_eta: N + N_gamma + N_eta + J_p] = x_c_star[N + Nc_gamma + Nc_eta: N + Nc_gamma + Nc_eta + J_p]  # Extracting J from constrained state x
        # Extracting coefficients from state:
        alpha_star = x_star[0: N]
        gamma_star = x_star[N: N + N_gamma]
        eta_star = x_star[N + N_gamma: N + N_gamma + N_eta]
        J_star = x_star[N + N_gamma + N_eta: N + N_gamma + N_eta + 1]
        # Computing F_star:
        F_star = F_alpha.eval(x_star, t_star) - np.matmul(J_alpha_star, alpha_star) - np.matmul(J_gamma_star, gamma_star) - np.matmul(J_eta_star, eta_star) - np.matmul(J_J_star, J_star)
        # Computing evolution operators for each coefficient:
        matrix_multiplier = np.linalg.inv(np.eye(N) - (dt / 2) * J_alpha_star)  # Computing matrix multiplier for evolution operators and additive vector
        F_evol_alpha = np.matmul(matrix_multiplier, (np.eye(N) + (dt / 2) * J_alpha_star))
        F_evol_gamma = np.matmul(matrix_multiplier, ((dt / 2) * J_gamma_star))
        F_evol_eta = np.matmul(matrix_multiplier, ((dt / 2) * J_eta_star))
        F_evol_J = np.matmul(matrix_multiplier, ((dt / 2) * J_J_star))
        # Updating evolution models to account for autoregressive models:
        F_evol_gamma = np.matmul(F_evol_gamma, B_gamma)
        F_evol_eta = np.matmul(F_evol_eta, B_eta)
        F_evol_J_tilde = np.matmul(F_evol_J, B_J)
        # Updating evolution models to account for continuity constraint:
        F_evol_gamma = np.matmul(F_evol_gamma, U_gamma_p)
        F_evol_eta = np.matmul(F_evol_eta, U_eta_p)
        # Computing evolution operator:
        F_evolution = np.zeros([N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p, N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p])  # Initialising
        F_evolution[0:N, 0:N] = F_evol_alpha
        F_evolution[0:N, N: N + gamma_p * Nc_gamma] = F_evol_gamma
        F_evolution[0:N, N + gamma_p * Nc_gamma: N + gamma_p * Nc_gamma + eta_p * Nc_eta] = F_evol_eta
        F_evolution[0:N, N + gamma_p * Nc_gamma + eta_p * Nc_eta: N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p] = F_evol_J_tilde
        # Row for gamma:
        F_evolution[N: N + gamma_p * Nc_gamma, N: N + gamma_p * Nc_gamma] = np.matmul(UT_gamma_p, np.matmul(A_gamma, U_gamma_p))
        # Row for eta:
        F_evolution[N + gamma_p * Nc_gamma: N + gamma_p * Nc_gamma + eta_p * Nc_eta, N + gamma_p * Nc_gamma: N + gamma_p * Nc_gamma + eta_p * Nc_eta] = np.matmul(UT_eta_p, np.matmul(A_eta, U_eta_p))
        # Row for J_tilde:
        F_evolution[N + gamma_p * Nc_gamma + eta_p * Nc_eta: N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p, N + gamma_p * Nc_gamma + eta_p * Nc_eta: N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p] = A_J
        # Computing evolution additive vector:
        b_evolution = np.zeros(N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p)  # Initialising
        b_evolution[0:N] = np.matmul(matrix_multiplier, (dt * F_star))
        return F_evolution, b_evolution

    # Function to compute evolution operator covariance:
    def compute_evolution_operator_alpha_covariance(Gamma_tilde_c_w, J_alpha_star, J_gamma_star, J_eta_star, J_J_star):
        # Getting current state covariance (i.e. accounting for VAR(p) model):
        Gamma_c_w = np.zeros([N + Nc_gamma + Nc_eta + J_p, N + Nc_gamma + Nc_eta + J_p])
        Gamma_c_w[0:N, 0:N] = Gamma_tilde_c_w[0:N, 0:N]
        Gamma_c_w[N: N + Nc_gamma, N: N + Nc_gamma] = Gamma_tilde_c_w[N: N + Nc_gamma, N: N + Nc_gamma]
        Gamma_c_w[N + Nc_gamma: N + Nc_gamma + Nc_eta, N + Nc_gamma: N + Nc_gamma + Nc_eta] = Gamma_tilde_c_w[N + gamma_p * Nc_gamma: N + gamma_p * Nc_gamma + Nc_eta, N + gamma_p * Nc_gamma: N + gamma_p * Nc_gamma + Nc_eta]
        Gamma_c_w[N + Nc_gamma + Nc_eta: N + Nc_gamma + Nc_eta + 1, N + Nc_gamma + Nc_eta: N + Nc_gamma + Nc_eta + 1] = Gamma_tilde_c_w[N + gamma_p * Nc_gamma + eta_p * Nc_eta: N + gamma_p * Nc_gamma + eta_p * Nc_eta + 1, N + gamma_p * Nc_gamma + eta_p * Nc_eta: N + gamma_p * Nc_gamma + eta_p * Nc_eta + 1]
        # Computing non-constrained state covariances:
        Gamma_alpha_w = Gamma_c_w[0:N, 0:N]
        Gamma_gamma_w = np.matmul(U_gamma, np.matmul(Gamma_c_w[N: N + Nc_gamma, N: N + Nc_gamma], UT_gamma))
        Gamma_eta_w = np.matmul(U_eta, np.matmul(Gamma_c_w[N + Nc_gamma: N + Nc_gamma + Nc_eta, N + Nc_gamma: N + Nc_gamma + Nc_eta], UT_eta))
        Gamma_J_w = Gamma_c_w[N + Nc_gamma + Nc_eta: N + Nc_gamma + Nc_eta + 1, N + Nc_gamma + Nc_eta: N + Nc_gamma + Nc_eta + 1]
        # Precomputations:
        matrix_multiplier = np.linalg.inv(np.eye(N) - (dt / 2) * J_alpha_star)
        Gamma_Jac_gamma = np.matmul(J_gamma_star, np.matmul(Gamma_gamma_w, np.transpose(J_gamma_star)))
        Gamma_Jac_eta = np.matmul(J_eta_star, np.matmul(Gamma_eta_w, np.transpose(J_eta_star)))
        Gamma_Jac_J = np.matmul(J_J_star, np.matmul(Gamma_J_w, np.transpose(J_J_star)))
        # Computing output:
        return Gamma_alpha_w + (dt ** 2 / 4) * np.matmul(matrix_multiplier, np.matmul(Gamma_Jac_gamma + Gamma_Jac_eta + Gamma_Jac_J, np.transpose(matrix_multiplier)))


    #######################################################
    # Constructing observation model:
    if use_DMPS_observation_model:
        M = N_channels  # Setting M to number of channels (i.e. observation dimensions to number of channels)
        DMA_transfer_function = get_DMA_transfer_function(R_inner, R_outer, length, Q_aerosol, Q_sheath, efficiency)  # Computes DMA transfer function
        H_alpha = compute_alpha_to_z_operator(F_alpha, DMA_transfer_function, N_channels, voltage_min, voltage_max)  # Computes operator for computing z(t) given alpha(t)
        H = np.zeros([M, N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p])  # Initialising
        H[0:M, 0:N] = H_alpha  # Observation operator
    else:
        M = len(Y)  # Dimension size of observations
        H_alpha = Size_distribution_observation_model(F_alpha, d_obs, M)  # Observation model
        H = np.zeros([M, N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p])  # Initialising
        H[0:M, 0:N] = H_alpha.H_phi  # Observation operator


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
    Gamma_alpha_w = basic_tools.compute_correlated_covariance_matrix(N, Np, Ne, sigma_alpha_w, sigma_alpha_w_correlation, use_element_multiplier=alpha_use_element_multipler)  # Covariance matrix computation
    Gamma_alpha_w[0: Np, 0: Np] = alpha_first_element_multiplier * Gamma_alpha_w[0: Np, 0: Np]  # First element multiplier

    # For gamma:
    sigma_gamma_w = np.array([sigma_gamma_w_0, sigma_gamma_w_1, sigma_gamma_w_2, sigma_gamma_w_3, sigma_gamma_w_4, sigma_gamma_w_5, sigma_gamma_w_6])  # Array of standard deviations
    Gamma_gamma_w = basic_tools.compute_correlated_covariance_matrix(N_gamma, Np_gamma, Ne_gamma, sigma_gamma_w, sigma_gamma_w_correlation, use_element_multiplier=gamma_use_element_multipler)  # Covariance matrix computation
    Gamma_gamma_w[0: Np_gamma, 0: Np_gamma] = gamma_first_element_multiplier * Gamma_gamma_w[0: Np_gamma, 0: Np_gamma]  # First element multiplier
    Gamma_gamma_tilde_w = np.zeros([gamma_p * N_gamma, gamma_p * N_gamma])  # Initialising covariance for gamma_tilde
    Gamma_gamma_tilde_w[0: N_gamma, 0: N_gamma] = Gamma_gamma_w  # Covariance for gamma_tilde
    Gamma_gamma_tilde_c_w = np.matmul(UT_gamma_p, np.matmul(Gamma_gamma_tilde_w, U_gamma_p))  # Continuity constraint conversion

    # For eta:
    sigma_eta_w = np.array([sigma_eta_w_0, sigma_eta_w_1, sigma_eta_w_2, sigma_eta_w_3, sigma_eta_w_4, sigma_eta_w_5, sigma_eta_w_6])  # Array of standard deviations
    Gamma_eta_w = basic_tools.compute_correlated_covariance_matrix(N_eta, Np_eta, Ne_eta, sigma_eta_w, sigma_eta_w_correlation, use_element_multiplier=eta_use_element_multipler)  # Covariance matrix computation
    Gamma_eta_w[0: Np_eta, 0: Np_eta] = eta_first_element_multiplier * Gamma_eta_w[0: Np_eta, 0: Np_eta]  # First element multiplier
    Gamma_eta_tilde_w = np.zeros([eta_p * N_eta, eta_p * N_eta])  # Initialising covariance for eta_tilde
    Gamma_eta_tilde_w[0: N_eta, 0: N_eta] = Gamma_eta_w  # Covariance for eta_tilde
    Gamma_eta_tilde_c_w = np.matmul(UT_eta_p, np.matmul(Gamma_eta_tilde_w, U_eta_p))  # Continuity constraint conversion

    # For J_tilde:
    Gamma_J_w = sigma_J_w ** 2
    Gamma_J_tilde_w = np.zeros([J_p, J_p])
    Gamma_J_tilde_w[0, 0] = Gamma_J_w

    # Assimilation:
    Gamma_tilde_c_w = np.zeros([N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p, N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p])  # Initialising noise covariance for state
    Gamma_tilde_c_w[0:N, 0:N] = Gamma_alpha_w  # Adding alpha covariance to state covariance
    Gamma_tilde_c_w[N: N + gamma_p * Nc_gamma, N: N + gamma_p * Nc_gamma] = Gamma_gamma_tilde_c_w  # Adding gamma covariance to state covariance
    Gamma_tilde_c_w[N + gamma_p * Nc_gamma: N + gamma_p * Nc_gamma + eta_p * Nc_eta, N + gamma_p * Nc_gamma: N + gamma_p * Nc_gamma + eta_p * Nc_eta] = Gamma_eta_tilde_c_w  # Adding eta covariance to state covariance
    Gamma_tilde_c_w[N + gamma_p * Nc_gamma + eta_p * Nc_eta: N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p, N + gamma_p * Nc_gamma + eta_p * Nc_eta: N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p] = Gamma_J_tilde_w  # Adding J_tilde covariance to state covariance


    #######################################################
    # Constructing prior and prior covariance:

    # For alpha:
    alpha_prior = F_alpha.compute_coefficients('alpha', initial_guess_size_distribution)  # Computing alpha coefficients from initial condition function
    sigma_alpha_prior = np.array([sigma_alpha_prior_0, sigma_alpha_prior_1, sigma_alpha_prior_2, sigma_alpha_prior_3, sigma_alpha_prior_4, sigma_alpha_prior_5, sigma_alpha_prior_6])  # Array of standard deviations
    Gamma_alpha_prior = basic_tools.compute_correlated_covariance_matrix(N, Np, Ne, sigma_alpha_prior, 0.001, use_element_multiplier=alpha_use_element_multipler)  # Covariance matrix computation
    Gamma_alpha_prior[0: Np, 0: Np] = alpha_first_element_multiplier * Gamma_alpha_prior[0: Np, 0: Np]  # First element multiplier

    # For gamma:
    gamma_prior = F_alpha.compute_coefficients('gamma', initial_guess_condensation_rate)    # Computing gamma coefficients from initial guess of condensation rate
    sigma_gamma_prior = np.array([sigma_gamma_prior_0, sigma_gamma_prior_1, sigma_gamma_prior_2, sigma_gamma_prior_3, sigma_gamma_prior_4, sigma_gamma_prior_5, sigma_gamma_prior_6])  # Array of standard deviations
    Gamma_gamma_prior = basic_tools.compute_correlated_covariance_matrix(N_gamma, Np_gamma, Ne_gamma, sigma_gamma_prior, 0.001, use_element_multiplier=gamma_use_element_multipler)  # Covariance matrix computation
    Gamma_gamma_prior[0: Np_gamma, 0: Np_gamma] = gamma_first_element_multiplier * Gamma_gamma_prior[0: Np_gamma, 0: Np_gamma]  # First element multiplier
    # Prior for gamma_tilde:
    gamma_tilde_prior = np.zeros([gamma_p * N_gamma])
    Gamma_gamma_tilde_prior = np.zeros([gamma_p * N_gamma, gamma_p * N_gamma])
    for i in range(gamma_p):
        gamma_tilde_prior[i * N_gamma: (i + 1) * N_gamma] = gamma_prior
        Gamma_gamma_tilde_prior[i * N_gamma: (i + 1) * N_gamma, i * N_gamma: (i + 1) * N_gamma] = Gamma_gamma_prior
    # Continuity constraint conversion:
    gamma_tilde_c_prior = np.matmul(UT_gamma_p, gamma_tilde_prior)
    Gamma_gamma_tilde_c_prior = np.matmul(UT_gamma_p, np.matmul(Gamma_gamma_tilde_prior, U_gamma_p))

    # For eta:
    eta_prior = F_alpha.compute_coefficients('eta', initial_guess_deposition_rate)    # Computing eta coefficients from initial guess of deposition rate
    sigma_eta_prior = np.array([sigma_eta_prior_0, sigma_eta_prior_1, sigma_eta_prior_2, sigma_eta_prior_3, sigma_eta_prior_4, sigma_eta_prior_5, sigma_eta_prior_6])  # Array of standard deviations
    Gamma_eta_prior = basic_tools.compute_correlated_covariance_matrix(N_eta, Np_eta, Ne_eta, sigma_eta_prior, 0.001, use_element_multiplier=eta_use_element_multipler)  # Covariance matrix computation
    Gamma_eta_prior[0: Np_eta, 0: Np_eta] = eta_first_element_multiplier * Gamma_eta_prior[0: Np_eta, 0: Np_eta]  # First element multiplier
    # Prior for eta_tilde:
    eta_tilde_prior = np.zeros([eta_p * N_eta])
    Gamma_eta_tilde_prior = np.zeros([eta_p * N_eta, eta_p * N_eta])
    for i in range(eta_p):
        eta_tilde_prior[i * N_eta: (i + 1) * N_eta] = eta_prior
        Gamma_eta_tilde_prior[i * N_eta: (i + 1) * N_eta, i * N_eta: (i + 1) * N_eta] = Gamma_eta_prior
    # Continuity constraint conversion:
    eta_tilde_c_prior = np.matmul(UT_eta_p, eta_tilde_prior)
    Gamma_eta_tilde_c_prior = np.matmul(UT_eta_p, np.matmul(Gamma_eta_tilde_prior, U_eta_p))

    # For J_tilde:
    Gamma_J_prior = sigma_J_prior ** 2
    Gamma_J_tilde_prior = np.zeros([J_p, J_p])
    for i in range(J_p):
        Gamma_J_tilde_prior[i, i] = Gamma_J_prior

    # Assimilation:
    x_tilde_c_prior = np.zeros([N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p])  # Initialising prior state
    x_tilde_c_prior[0:N] = alpha_prior  # Adding alpha prior to prior state
    x_tilde_c_prior[N: N + gamma_p * Nc_gamma] = gamma_tilde_c_prior  # Adding gamma prior to prior state
    x_tilde_c_prior[N + gamma_p * Nc_gamma: N + gamma_p * Nc_gamma + eta_p * Nc_eta] = eta_tilde_c_prior  # Adding eta prior to prior state
    Gamma_tilde_c_prior = np.zeros([N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p, N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p])  # Initialising prior covariance for state
    Gamma_tilde_c_prior[0:N, 0:N] = Gamma_alpha_prior  # Adding alpha covariance to state covariance
    Gamma_tilde_c_prior[N: N + gamma_p * Nc_gamma, N: N + gamma_p * Nc_gamma] = Gamma_gamma_tilde_c_prior  # Adding gamma covariance to state covariance
    Gamma_tilde_c_prior[N + gamma_p * Nc_gamma: N + gamma_p * Nc_gamma + eta_p * Nc_eta, N + gamma_p * Nc_gamma: N + gamma_p * Nc_gamma + eta_p * Nc_eta] = Gamma_eta_tilde_c_prior  # Adding eta covariance to state covariance
    Gamma_tilde_c_prior[N + gamma_p * Nc_gamma + eta_p * Nc_eta: N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p, N + gamma_p * Nc_gamma + eta_p * Nc_eta: N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p] = Gamma_J_tilde_prior  # Adding J_tilde covariance to state covariance


    #######################################################
    # Initialising state and adding prior:
    x_tilde_c = np.zeros([N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p, NT])  # Initialising state = [x_0, x_1, ..., x_{NT - 1}]
    Gamma_tilde_c = np.zeros([NT, N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p, N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p])  # Initialising state covariance matrix Gamma_0, Gamma_1, ..., Gamma_NT
    x_tilde_c_predict = np.zeros([N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p, NT])  # Initialising predicted state
    Gamma_tilde_c_predict = np.zeros([NT, N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p, N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p])  # Initialising predicted state covariance
    x_tilde_c[:, 0], x_tilde_c_predict[:, 0] = x_tilde_c_prior, x_tilde_c_prior  # Adding prior to states
    Gamma_tilde_c[0], Gamma_tilde_c_predict[0] = Gamma_tilde_c_prior, Gamma_tilde_c_prior  # Adding prior to state covariance matrices


    #######################################################
    # Initialising evolution operator and additive evolution vector:
    F = np.zeros([NT, N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p, N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p])  # Initialising evolution operator F_0, F_1, ..., F_{NT - 1}
    b = np.zeros([N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p, NT])  # Initialising additive evolution vector b_0, b_1, ..., b_{NT - 1}


    #######################################################
    # Constructing extended Kalman filter model:
    model = Kalman_filter(F[0], H, Gamma_tilde_c_w, Gamma_v, NT, additive_evolution_vector=b[:, 0])


    #######################################################
    # Computing time evolution of model:
    print('Computing Kalman filter estimates...')
    t = np.zeros(NT)  # Initialising time array
    for k in tqdm(range(NT - 1)):  # Iterating over time
        J_alpha_star, J_gamma_star, J_eta_star, J_J_star = compute_evolution_operator_Jacobians(x_tilde_c[:, k], t[k])  # Computing evolution operator Jacobian
        F[k], b[:, k] = compute_evolution_operator(x_tilde_c[:, k], t[k], J_alpha_star, J_gamma_star, J_eta_star, J_J_star)  # Computing evolution operator F and vector b
        model.F, model.additive_evolution_vector = F[k], b[:, k]  # Adding updated evolution operator and vector b to Kalman Filter
        model.Gamma_w[0:N, 0:N] = compute_evolution_operator_alpha_covariance(model.Gamma_w, J_alpha_star, J_gamma_star, J_eta_star, J_J_star)  # Computing evolution model covariance matrix for alpha coefficients
        x_tilde_c_predict[:, k + 1], Gamma_tilde_c_predict[k + 1] = model.predict(x_tilde_c[:, k], Gamma_tilde_c[k], k)  # Computing prediction
        x_tilde_c[:, k + 1], Gamma_tilde_c[k + 1] = model.update(x_tilde_c_predict[:, k + 1], Gamma_tilde_c_predict[k + 1], Y[:, k + 1], k)  # Computing update
        t[k + 1] = (k + 1) * dt  # Time (hours)


    #######################################################
    # Computing smoothed estimates:
    if smoothing:
        print('Computing Kalman smoother estimates...')
        x_tilde_c, Gamma_tilde_c = compute_fixed_interval_Kalman_smoother(F, NT, N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p, x_tilde_c, Gamma_tilde_c, x_tilde_c_predict, Gamma_tilde_c_predict)


    #######################################################
    # Computing single state from multiple state vector (computing x_c from x_tilde_c):
    x_c = np.zeros([N + Nc_gamma + Nc_eta + 1, NT])
    Gamma_c = np.zeros([NT, N + Nc_gamma + Nc_eta + 1, N + Nc_gamma + Nc_eta + 1])
    for k in range(NT):
        x_c[0: N, k] = x_tilde_c[0: N, k]
        x_c[N: N + Nc_gamma, k] = x_tilde_c[N: N + Nc_gamma, k]
        x_c[N + Nc_gamma: N + Nc_gamma + Nc_eta, k] = x_tilde_c[N + gamma_p * Nc_gamma: N + gamma_p * Nc_gamma + Nc_eta, k]
        x_c[N + Nc_gamma + Nc_eta, k] = x_tilde_c[N + gamma_p * Nc_gamma + eta_p * Nc_eta, k]
        Gamma_c[k, 0: N, 0: N] = Gamma_tilde_c[k, 0: N, 0: N]
        Gamma_c[k, N: N + Nc_gamma, N: N + Nc_gamma] = Gamma_tilde_c[k, N: N + Nc_gamma, N: N + Nc_gamma]
        Gamma_c[k, N + Nc_gamma: N + Nc_gamma + Nc_eta, N + Nc_gamma: N + Nc_gamma + Nc_eta] = Gamma_tilde_c[k, N + gamma_p * Nc_gamma: N + gamma_p * Nc_gamma + Nc_eta, N + gamma_p * Nc_gamma: N + gamma_p * Nc_gamma + Nc_eta]
        Gamma_c[k, N + Nc_gamma + Nc_eta, N + Nc_gamma + Nc_eta] = Gamma_tilde_c[k, N + gamma_p * Nc_gamma + eta_p * Nc_eta, N + gamma_p * Nc_gamma + eta_p * Nc_eta]


    #######################################################
    # Computing unconstrained state:
    x = np.zeros([N + N_gamma + N_eta + 1, NT])  # Initialising state = [x_0, x_1, ..., x_{NT - 1}]
    Gamma = np.zeros([NT, N + N_gamma + N_eta + 1, N + N_gamma + N_eta + 1])  # Initialising state covariance matrix Gamma_0, Gamma_1, ..., Gamma_NT
    for k in range(NT):
        x[0: N, k] = x_c[0: N, k]  # Extracting alpha from constrained state x
        x[N: N + N_gamma, k] = np.matmul(U_gamma, x_c[N: N + Nc_gamma, k])  # Extracting gamma from constrained state x
        x[N + N_gamma: N + N_gamma + N_eta, k] = np.matmul(U_eta, x_c[N + Nc_gamma: N + Nc_gamma + Nc_eta, k])  # Extracting eta from constrained state x
        x[N + N_gamma + N_eta, k] = x_c[N + Nc_gamma + Nc_eta, k]  # Extracting J from constrained state x
        Gamma[k, 0: N, 0: N] = Gamma_c[k, 0: N, 0: N]
        Gamma[k, N: N + N_gamma, N: N + N_gamma] = np.matmul(U_gamma, np.matmul(Gamma_c[k, N: N + Nc_gamma, N: N + Nc_gamma], UT_gamma))
        Gamma[k, N + N_gamma: N + N_gamma + N_eta, N + N_gamma: N + N_gamma + N_eta] = np.matmul(U_eta, np.matmul(Gamma_c[k, N + Nc_gamma: N + Nc_gamma + Nc_eta, N + Nc_gamma: N + Nc_gamma + Nc_eta], UT_eta))
        Gamma[k, N + N_gamma + N_eta, N + N_gamma + N_eta] = Gamma_c[k, N + Nc_gamma + Nc_eta, N + Nc_gamma + Nc_eta]


    #######################################################
    # Extracting alpha, gamma, eta, and covariances from state:
    alpha = x[0:N, :]  # Size distribution coefficients
    gamma = x[N: N + N_gamma, :]  # Condensation rate coefficients
    eta = x[N + N_gamma: N + N_gamma + N_eta, :]  # Deposition rate coefficients
    J = x[N + N_gamma + N_eta, :]  # Nucleation rate coefficients
    Gamma_alpha = Gamma[:, 0:N, 0:N]  # Size distribution covariance
    Gamma_gamma = Gamma[:, N: N + N_gamma, N: N + N_gamma]  # Condensation rate covariance
    Gamma_eta = Gamma[:, N + N_gamma: N + N_gamma + N_eta, N + N_gamma: N + N_gamma + N_eta]  # Deposition rate covariance
    Gamma_J = Gamma[:, N + N_gamma + N_eta, N + N_gamma + N_eta]  # Nucleation rate covariance


    #######################################################
    # Computing plotting discretisation:
    # Size distribution:
    d_plot, v_plot, n_logDp_plot, sigma_n_logDp = F_alpha.get_nplot_discretisation(alpha, Gamma_alpha=Gamma_alpha, convert_x_to_logDp=True)
    n_logDp_plot_upper = n_logDp_plot + 2 * sigma_n_logDp
    n_logDp_plot_lower = n_logDp_plot - 2 * sigma_n_logDp
    # Condensation rate:
    _, log_d_plot_cond, cond_Dp_plot, sigma_cond_Dp = F_alpha.get_parameter_estimation_discretisation('condensation', gamma, Gamma_gamma)
    cond_Dp_plot_upper = cond_Dp_plot + 2 * sigma_cond_Dp
    cond_Dp_plot_lower = cond_Dp_plot - 2 * sigma_cond_Dp
    # Deposition rate:
    _, log_d_plot_depo, depo_plot, sigma_depo = F_alpha.get_parameter_estimation_discretisation('deposition', eta, Gamma_eta)
    depo_plot_upper = depo_plot + 2 * sigma_depo
    depo_plot_lower = depo_plot - 2 * sigma_depo
    # Nucleation rate:
    J_logDp_plot, sigma_J_logDp = F_alpha.get_parameter_estimation_discretisation('nucleation', J, Gamma_J, convert_x_to_logDp=True)
    J_logDp_plot_upper = J_logDp_plot + 2 * sigma_J_logDp
    J_logDp_plot_lower = J_logDp_plot - 2 * sigma_J_logDp


    #######################################################
    # Computing true underlying parameters plotting discretisation:
    Nplot_cond = len(log_d_plot_cond)  # Length of size discretisation
    Nplot_depo = len(log_d_plot_depo)  # Length of size discretisation
    d_plot_cond = np.exp(log_d_plot_cond)  # Computing Dp plotting discretisation
    d_plot_depo = np.exp(log_d_plot_depo)  # Computing Dp plotting discretisation
    cond_Dp_true_plot = np.zeros([Nplot_cond, NT])  # Initialising condensation rate
    depo_true_plot = np.zeros([Nplot_depo, NT])  # Initialising deposition rate
    sorc_x_true_plot = np.zeros(NT)  # Initialising ln(volume)-based source (nucleation) rate
    for k in range(NT):
        sorc_x_true_plot[k] = sorc(t[k])  # Computing ln(volume)-based nucleation rate
        for i in range(Nplot_cond):
            cond_Dp_true_plot[i, k] = cond(d_plot_cond[i])  # Computing condensation rate
        for i in range(Nplot_depo):
            depo_true_plot[i, k] = depo(d_plot_depo[i])  # Computing deposition rate
    sorc_logDp_true_plot = change_basis_x_to_logDp_sorc(sorc_x_true_plot, vmin, Dp_min)  # Computing log10(diameter)-based nucleation rate


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
    xticks = [0.01, 0.1, 1]  # Plot x-tick labels
    xlimits = [0.01, 1]  # Plot boundary limits for x-axis
    ylimits = [0, 10000]  # Plot boundary limits for y-axis
    xlabel = '$D_p$ ($\mu$m)'  # x-axis label for 1D animation plot
    ylabel = '$\dfrac{dN}{dlogD_p}$ (cm$^{-3})$'  # y-axis label for 1D animation plot
    title = 'Size distribution estimation'  # Title for 1D animation plot
    legend = ['Estimate', '$\pm 2 \sigma$', '', 'Truth']  # Adding legend to plot
    line_color = ['blue', 'blue', 'blue', 'green']  # Colors of lines in plot
    line_style = ['solid', 'dashed', 'dashed', 'solid']  # Style of lines in plot
    time = t  # Array where time[i] is plotted (and animated)
    timetext = ('Time = ', ' hours')  # Tuple where text to be animated is: timetext[0] + 'time[i]' + timetext[1]
    delay = 30  # Delay between frames in milliseconds

    # Parameters for condensation plot:
    yscale_cond = 'linear'  # y-axis scaling ('linear' or 'log')
    ylimits_cond = [0, 0.12]  # Plot boundary limits for y-axis
    xlabel_cond = '$D_p$ ($\mu$m)'  # x-axis label for plot
    ylabel_cond = '$I(D_p)$ ($\mu$m hour$^{-1}$)'  # y-axis label for plot
    title_cond = 'Condensation rate estimation'  # Title for plot
    location_cond = location + '2'  # Location for plot
    legend_cond = ['Estimate', '$\pm 2 \sigma$', '', 'Truth']  # Adding legend to plot
    line_color_cond = ['blue', 'blue', 'blue', 'green']  # Colors of lines in plot
    line_style_cond = ['solid', 'dashed', 'dashed', 'solid']  # Style of lines in plot
    delay_cond = 10  # Delay between frames in milliseconds

    # Parameters for deposition plot:
    yscale_depo = 'linear'  # y-axis scaling ('linear' or 'log')
    ylimits_depo = [0, 0.4]  # Plot boundary limits for y-axis
    xlabel_depo = '$D_p$ ($\mu$m)'  # x-axis label for plot
    ylabel_depo = '$d(D_p)$ (hour$^{-1}$)'  # y-axis label for plot
    title_depo = 'Deposition rate estimation'  # Title for plot
    location_depo = location + '3'  # Location for plot
    legend_depo = ['Estimate', '$\pm 2 \sigma$', '', 'Truth']  # Adding legend to plot
    line_color_depo = ['blue', 'blue', 'blue', 'green']  # Colors of lines in plot
    line_style_depo = ['solid', 'dashed', 'dashed', 'solid']  # Style of lines in plot
    delay_depo = delay_cond  # Delay between frames in milliseconds

    # Size distribution animation:
    basic_tools.plot_1D_animation(d_plot, n_logDp_plot, n_logDp_plot_lower, n_logDp_plot_upper, plot_add=(d_true, n_logDp_true), xticks=xticks, xlimits=xlimits, ylimits=ylimits, xscale=xscale, xlabel=xlabel, ylabel=ylabel, title=title,
                                  delay=delay, location=location, legend=legend, time=time, timetext=timetext, line_color=line_color, line_style=line_style, doing_mainloop=False)

    # Condensation rate animation:
    basic_tools.plot_1D_animation(d_plot_cond, cond_Dp_plot, cond_Dp_plot_lower, cond_Dp_plot_upper, cond_Dp_true_plot, xticks=xticks, xlimits=xlimits, xscale=xscale, xlabel=xlabel_cond, ylabel=ylabel_cond, title=title_cond,
                                  delay=delay_cond, ylimits=ylimits_cond, yscale=yscale_cond, location=location_cond, legend=legend_cond, time=time, timetext=timetext, line_color=line_color_cond, line_style=line_style_cond, doing_mainloop=False)

    # Deposition rate animation:
    basic_tools.plot_1D_animation(d_plot_depo, depo_plot, depo_plot_lower, depo_plot_upper, depo_true_plot, xticks=xticks, xlimits=xlimits, xscale=xscale, xlabel=xlabel_depo, ylabel=ylabel_depo, title=title_depo,
                                  delay=delay_depo, ylimits=ylimits_depo, yscale=yscale_depo, location=location_depo, legend=legend_depo, time=time, timetext=timetext, line_color=line_color_depo, line_style=line_style_depo, doing_mainloop=False)

    # Mainloop and print:
    if plot_animations:
        print('Plotting animations...')
        mainloop()  # Runs tkinter GUI for plots and animations


    #######################################################
    # Plotting nucleation rate:
    if plot_nucleation:
        print('Plotting nucleation...')
        figJ, axJ = plt.subplots(figsize=(8.00, 5.00), dpi=100)
        plt.plot(time, J_logDp_plot, 'b-', label='Estimate')
        plt.plot(time, J_logDp_plot_lower, 'b--', label='$\pm 2 \sigma$')
        plt.plot(time, J_logDp_plot_upper, 'b--')
        plt.plot(time, sorc_logDp_true_plot, 'g-', label='Truth')
        axJ.set_xlim([0, T])
        axJ.set_ylim([-3000, 12000])
        axJ.set_xlabel('$t$ (hour)', fontsize=12)
        axJ.set_ylabel('$J(t)$ \n (cm$^{-3}$ hour$^{-1}$)', fontsize=12, rotation=0)
        axJ.yaxis.set_label_coords(-0.015, 1.02)
        axJ.set_title('Nucleation rate estimation', fontsize=12)
        axJ.grid()
        axJ.legend(fontsize=11, loc='upper left')


    #######################################################
    # Images:

    # Parameters for size distribution images:
    yscale_image = 'log'  # Change scale of y-axis (linear or log)
    yticks_image = xticks  # Plot y-tick labels
    xlabel_image = 'Time (hours)'  # x-axis label for image
    ylabel_image = '$D_p$ ($\mu$m) \n'  # y-axis label for image
    ylabelcoords = (-0.06, 0.96)  # y-axis label coordinates
    title_image = 'Size distribution estimation'  # Title for image
    title_image_observations = 'Simulated observations'  # Title for image
    image_min = 10  # Minimum of image colour
    image_max = 10000  # Maximum of image colour
    cmap = 'jet'  # Colour map of image
    cbarlabel = '$\dfrac{dN}{dlogD_p}$ (cm$^{-3})$'  # Label of colour bar
    cbarticks = [10, 100, 1000, 10000]  # Ticks of colorbar

    # Plotting images:
    if plot_images:
        print('Plotting images...')
        basic_tools.image_plot(time, d_plot, n_logDp_plot, ylimits=xlimits, xlabel=xlabel_image, ylabel=ylabel_image, title=title_image,
                               yscale=yscale_image, ylabelcoords=ylabelcoords, yticks=yticks_image, image_min=image_min, image_max=image_max, cmap=cmap, cbarlabel=cbarlabel, cbarticks=cbarticks)
        basic_tools.image_plot(time, d_obs, Y, ylimits=xlimits, xlabel=xlabel_image, ylabel=ylabel_image, title=title_image_observations,
                               yscale=yscale_image, ylabelcoords=ylabelcoords, yticks=yticks_image, image_min=image_min, image_max=image_max, cmap=cmap, cbarlabel=cbarlabel, cbarticks=cbarticks)


    # Final print statements
    basic_tools.print_lines()  # Print lines in console
    print()  # Print space in console

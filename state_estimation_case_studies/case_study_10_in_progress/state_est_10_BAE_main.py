"""

Title: Computes Bayesian approximation error for evolution model of the particle size distribution, condensation rate, deposition rate, and nucleation rate
Author: Vincent Russell
Date: May 5, 2022

"""


#######################################################
# Modules:
import numpy as np
import time as tm
from tqdm import tqdm

# Local modules:
import basic_tools
from evolution_models.tools import GDE_evolution_model, GDE_Jacobian, compute_U


#######################################################
# Importing parameter file:
from state_estimation_case_studies.case_study_10_in_progress.state_est_10_BAE_parameters import *


#######################################################
if __name__ == '__main__':

    #######################################################
    # Initialising timer for total computation:
    initial_time = tm.time()  # Initial time stamp


    #######################################################
    # Constructing size distribution evolution model:
    F_alpha = GDE_evolution_model(Ne, Np, vmin, vmax, dt, NT, boundary_zero=boundary_zero)  # Initialising evolution model
    F_alpha.add_unknown('condensation', Ne_gamma, Np_gamma)  # Adding condensation as unknown to evolution model
    F_alpha.add_unknown('deposition', Ne_eta, Np_eta)  # Adding deposition as unknown to evolution model
    F_alpha.add_unknown('source')  # Adding source as unknown to evolution model
    F_alpha.add_process('coagulation', coag, load_coagulation=load_coagulation)  # Adding coagulation to evolution model
    F_alpha.compile()  # Compiling evolution model


    #######################################################
    # Constructing reduced size distribution evolution model:
    F_alpha_r = GDE_evolution_model(Ne_r, Np_r, vmin, vmax, dt, NT, boundary_zero=boundary_zero)  # Initialising evolution model
    F_alpha_r.add_unknown('condensation', Ne_gamma_r, Np_gamma_r)  # Adding condensation as unknown to evolution model
    F_alpha_r.add_unknown('deposition', Ne_eta_r, Np_eta_r)  # Adding deposition as unknown to evolution model
    F_alpha_r.add_unknown('source')  # Adding source as unknown to evolution model
    F_alpha_r.compile()  # Compiling evolution model


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
    # Constructing prior and prior covariance:

    # For alpha:
    alpha_prior = F_alpha.compute_coefficients('alpha', initial_guess_size_distribution)  # Computing alpha coefficients from initial condition function

    # For gamma:
    gamma_prior = F_alpha.compute_coefficients('gamma', initial_guess_condensation_rate)    # Computing gamma coefficients from initial guess of condensation rate
    # Prior for gamma_tilde:
    gamma_tilde_prior = np.zeros([gamma_p * N_gamma])
    for i in range(gamma_p):
        gamma_tilde_prior[i * N_gamma: (i + 1) * N_gamma] = gamma_prior
    # Continuity constraint conversion:
    gamma_tilde_c_prior = np.matmul(UT_gamma_p, gamma_tilde_prior)

    # For eta:
    eta_prior = F_alpha.compute_coefficients('eta', initial_guess_deposition_rate)    # Computing eta coefficients from initial guess of deposition rate
    # Prior for eta_tilde:
    eta_tilde_prior = np.zeros([eta_p * N_eta])
    for i in range(eta_p):
        eta_tilde_prior[i * N_eta: (i + 1) * N_eta] = eta_prior
    # Continuity constraint conversion:
    eta_tilde_c_prior = np.matmul(UT_eta_p, eta_tilde_prior)

    # Assimilation:
    x_tilde_c_prior = np.zeros([N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p])  # Initialising prior state
    x_tilde_c_prior[0:N] = alpha_prior  # Adding alpha prior to prior state
    x_tilde_c_prior[N: N + gamma_p * Nc_gamma] = gamma_tilde_c_prior  # Adding gamma prior to prior state
    x_tilde_c_prior[N + gamma_p * Nc_gamma: N + gamma_p * Nc_gamma + eta_p * Nc_eta] = eta_tilde_c_prior  # Adding eta prior to prior state


    #######################################################
    # Initialising state and adding prior:
    x_tilde_c = np.zeros([N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p, NT])  # Initialising state = [x_0, x_1, ..., x_{NT - 1}]
    x_tilde_c[:, 0] = x_tilde_c_prior  # Adding prior to state


    #######################################################
    # Initialising evolution operator and additive evolution vector:
    F = np.zeros([NT, N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p, N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p])  # Initialising evolution operator F_0, F_1, ..., F_{NT - 1}
    b = np.zeros([N + gamma_p * Nc_gamma + eta_p * Nc_eta + J_p, NT])  # Initialising additive evolution vector b_0, b_1, ..., b_{NT - 1}


    #######################################################
    # Computing time evolution of model:
    for k in tqdm(range(NT - 1)):  # Iterating over time
        J_alpha_star, J_gamma_star, J_eta_star, J_J_star = compute_evolution_operator_Jacobians(x_tilde_c[:, k], t[k])  # Computing evolution operator Jacobian
        F[k], b[:, k] = compute_evolution_operator(x_tilde_c[:, k], t[k], J_alpha_star, J_gamma_star, J_eta_star, J_J_star)  # Computing evolution operator F and vector b


    #######################################################
    # Printing total computation time:
    computation_time = round(tm.time() - initial_time, 3)  # Initial time stamp
    print('Total computation time:', str(computation_time), 'seconds.')  # Print statement


    # Final print statements
    basic_tools.print_lines()  # Print lines in console
    print()  # Print space in console

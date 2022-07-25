"""

Title: BAE computation for the state estimation of aerosol particle size distribution
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
from evolution_models.tools import GDE_evolution_model, GDE_Jacobian, compute_U, compute_G


#######################################################
# Importing parameter file:
from state_identification_case_studies.case_study_07_0.state_iden_07_0_BAE_computation_parameters import *


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
    # Constructing size distribution evolution model:
    F_alpha = GDE_evolution_model(Ne, Np, vmin, vmax, dt, NT, boundary_zero=boundary_zero, print_status=False)  # Initialising evolution model
    F_alpha.add_unknown('condensation', Ne_gamma, Np_gamma)  # Adding condensation as unknown to evolution model
    F_alpha.add_unknown('source')  # Adding source as unknown to evolution model
    F_alpha.add_process('coagulation', coag, load_coagulation=load_coagulation, coagulation_suffix=coagulation_suffix)  # Adding coagulation to evolution model


    #######################################################
    # Constructing reduced size distribution evolution model:
    F_alpha_r = GDE_evolution_model(Ne_r, Np_r, vmin, vmax, dt, NT, boundary_zero=boundary_zero, print_status=False)  # Initialising reduced evolution model
    F_alpha_r.add_unknown('condensation', Ne_gamma, Np_gamma)  # Adding condensation as unknown to evolution model
    F_alpha_r.add_process('deposition', guess_depo)  # Adding deposition to evolution model
    F_alpha_r.add_unknown('source')  # Adding source as unknown to evolution model
    F_alpha_r.compile()  # Compiling evolution model


    #######################################################
    # Continuity constraint computations for gamma (condensation):
    num_constraints_gamma = Ne_gamma - 1  # Number of contraints
    Nc_gamma = N_gamma - num_constraints_gamma  # Dimensions of constrained gamma
    U_gamma, UT_gamma = compute_U(N_gamma, Ne_gamma, Np_gamma, F_alpha_r.phi_gamma, F_alpha_r.x_boundaries_gamma)  # Computing null space continuity matrix
    # Continuity constraint matrix with multiple states in time (accounting for VAR(p) model):
    U_gamma_p = np.zeros([gamma_p * N_gamma, gamma_p * Nc_gamma])  # Initialising
    UT_gamma_p = np.zeros([gamma_p * Nc_gamma, gamma_p * N_gamma])  # Initialising
    for i in range(gamma_p):
        U_gamma_p[i * N_gamma: (i + 1) * N_gamma, i * Nc_gamma: (i + 1) * Nc_gamma] = U_gamma
        UT_gamma_p[i * Nc_gamma: (i + 1) * Nc_gamma, i * N_gamma: (i + 1) * N_gamma] = UT_gamma


    #######################################################
    # Initialising approximation error, and computing projection matrix:

    # Initialising approximation error:
    epsilon = np.zeros([N_iterations, N_r + gamma_p * Nc_gamma + J_p, NT])  # Initialising BAE

    # For alpha:
    G_alpha = compute_G(N, N_r, Np, Np_r, F_alpha.phi, F_alpha_r.phi, F_alpha.x_boundaries)  # Computing matrix G for projection operator
    P_alpha = np.matmul(np.linalg.inv(F_alpha_r.M), G_alpha)  # Computing projection operator

    # For gamma (assuming N_r = N):
    P_gamma = np.eye(gamma_p * Nc_gamma)

    # For J (assuming N_r = N):
    P_J = np.eye(J_p)

    # Assimilation:
    P = np.zeros([N_r + gamma_p * Nc_gamma + J_p, N + gamma_p * Nc_gamma + J_p])  # Initialising projection operator
    P[0:N_r, 0:N] = P_alpha  # Adding alpha projection
    P[N_r: N_r + gamma_p * Nc_gamma, N: N + gamma_p * Nc_gamma] = P_gamma  # Adding gamma projection
    P[N_r + gamma_p * Nc_gamma: N_r + gamma_p * Nc_gamma + J_p, N + gamma_p * Nc_gamma: N + gamma_p * Nc_gamma + J_p] = P_J  # Adding J projection


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
    J_F_alpha_r = GDE_Jacobian(F_alpha_r)  # Evolution Jacobian
    dF_alpha_d_alpha_r = J_F_alpha_r.eval_d_alpha  # Derivative with respect to alpha
    dF_alpha_d_gamma_r = J_F_alpha_r.eval_d_gamma  # Derivative with respect to gamma
    dF_alpha_d_J_r = J_F_alpha_r.eval_d_J  # Derivative with respect to J


    #######################################################
    # Functions to compute/update reduced evolution operator using Crank-Nicolson method:

    # Function to compute evolution operator Jacobians:
    def compute_evolution_operator_Jacobians_r(x_tilde_c_star_r, t_star):
        # Getting current state (i.e. accounting for VAR(p) model):
        x_c_star_r = np.zeros(N_r + Nc_gamma + J_p)
        x_c_star_r[0: N_r] = x_tilde_c_star_r[0: N_r]
        x_c_star_r[N_r: N_r + Nc_gamma] = np.matmul(C_gamma, x_tilde_c_star_r[N_r: N_r + gamma_p * Nc_gamma])
        x_c_star_r[N_r + Nc_gamma: N_r + Nc_gamma + J_p] = x_tilde_c_star_r[N_r + gamma_p * Nc_gamma: N_r + gamma_p * Nc_gamma + J_p]
        # Computing non-constrained state:
        x_star_r = np.zeros(N_r + N_gamma + J_p)  # Initialising
        x_star_r[0: N_r] = x_c_star_r[0: N_r]  # Extracting alpha from constrained state x
        x_star_r[N_r: N_r + N_gamma] = np.matmul(U_gamma, x_c_star_r[N_r: N_r + Nc_gamma])  # Extracting gamma from constrained state x
        x_star_r[N_r + N_gamma: N_r + N_gamma + J_p] = x_c_star_r[N_r + Nc_gamma: N_r + Nc_gamma + J_p]  # Extracting J from constrained state x
        # Computing Jacobians:
        J_alpha_star_r = dF_alpha_d_alpha_r(x_star_r, t_star)
        J_gamma_star_r = dF_alpha_d_gamma_r(x_star_r, t_star)
        J_J_star_r = dF_alpha_d_J_r(x_star_r, t_star)
        return J_alpha_star_r, J_gamma_star_r, J_J_star_r

    # Function to compute evolution operator:
    def compute_evolution_operator_r(x_tilde_c_star_r, t_star, J_alpha_star_r, J_gamma_star_r, J_J_star_r):
        # Getting current state (i.e. accounting for VAR(p) model):
        x_c_star_r = np.zeros(N_r + Nc_gamma + J_p)
        x_c_star_r[0: N_r] = x_tilde_c_star_r[0: N_r]
        x_c_star_r[N_r: N_r + Nc_gamma] = np.matmul(C_gamma, x_tilde_c_star_r[N_r: N_r + gamma_p * Nc_gamma])
        x_c_star_r[N_r + Nc_gamma: N_r + Nc_gamma + J_p] = x_tilde_c_star_r[N_r + gamma_p * Nc_gamma: N_r + gamma_p * Nc_gamma + J_p]
        # Computing non-constrained state:
        x_star_r = np.zeros(N_r + N_gamma + J_p)  # Initialising
        x_star_r[0: N_r] = x_c_star_r[0: N_r]  # Extracting alpha from constrained state x
        x_star_r[N_r: N_r + N_gamma] = np.matmul(U_gamma, x_c_star_r[N_r: N_r + Nc_gamma])  # Extracting gamma from constrained state x
        x_star_r[N_r + N_gamma: N_r + N_gamma + J_p] = x_c_star_r[N_r + Nc_gamma: N_r + Nc_gamma + J_p]  # Extracting J from constrained state x
        # Extracting coefficients from state:
        alpha_star_r = x_star_r[0: N_r]
        gamma_star_r = x_star_r[N_r: N_r + N_gamma]
        J_star_r = x_star_r[N_r + N_gamma: N_r + N_gamma + 1]
        # Computing F_star:
        F_star_r = F_alpha_r.eval(x_star_r, t_star) - np.matmul(J_alpha_star_r, alpha_star_r) - np.matmul(J_gamma_star_r, gamma_star_r) - np.matmul(J_J_star_r, J_star_r)
        # Computing evolution operators for each coefficient:
        matrix_multiplier_r = np.linalg.inv(np.eye(N_r) - (dt / 2) * J_alpha_star_r)  # Computing matrix multiplier for evolution operators and additive vector
        F_evol_alpha_r = np.matmul(matrix_multiplier_r, (np.eye(N_r) + (dt / 2) * J_alpha_star_r))
        F_evol_gamma_r = np.matmul(matrix_multiplier_r, ((dt / 2) * J_gamma_star_r))
        F_evol_J_r = np.matmul(matrix_multiplier_r, ((dt / 2) * J_J_star_r))
        # Updating evolution models to account for autoregressive models:
        F_evol_gamma_r = np.matmul(F_evol_gamma_r, B_gamma)
        F_evol_J_tilde_r = np.matmul(F_evol_J_r, B_J)
        # Updating evolution models to account for continuity constraint:
        F_evol_gamma_r = np.matmul(F_evol_gamma_r, U_gamma_p)
        # Computing evolution operator:
        F_evolution_r = np.zeros([N_r + gamma_p * Nc_gamma + J_p, N_r + gamma_p * Nc_gamma + J_p])  # Initialising
        F_evolution_r[0:N_r, 0:N_r] = F_evol_alpha_r
        F_evolution_r[0:N_r, N_r: N_r + gamma_p * Nc_gamma] = F_evol_gamma_r
        F_evolution_r[0:N_r, N_r + gamma_p * Nc_gamma: N_r + gamma_p * Nc_gamma + J_p] = F_evol_J_tilde_r
        # Row for gamma:
        F_evolution_r[N_r: N_r + gamma_p * Nc_gamma, N_r: N_r + gamma_p * Nc_gamma] = np.matmul(UT_gamma_p, np.matmul(A_gamma, U_gamma_p))
        # Row for J_tilde:
        F_evolution_r[N_r + gamma_p * Nc_gamma: N_r + gamma_p * Nc_gamma + J_p, N_r + gamma_p * Nc_gamma: N_r + gamma_p * Nc_gamma + J_p] = A_J
        # Computing evolution additive vector:
        b_evolution_r = np.zeros(N_r + gamma_p * Nc_gamma + J_p)  # Initialising
        b_evolution_r[0:N_r] = np.matmul(matrix_multiplier_r, (dt * F_star_r))
        return F_evolution_r, b_evolution_r


    #######################################################
    # Constructing prior:

    # For alpha:
    alpha_prior_r = F_alpha_r.compute_coefficients('alpha', initial_guess_size_distribution)  # Computing alpha coefficients from initial condition function

    # For gamma:
    gamma_prior_r = F_alpha_r.compute_coefficients('gamma', initial_guess_condensation_rate)    # Computing gamma coefficients from initial guess of condensation rate
    # Prior for gamma_tilde:
    gamma_tilde_prior_r = np.zeros([gamma_p * N_gamma])
    for i in range(gamma_p):
        gamma_tilde_prior_r[i * N_gamma: (i + 1) * N_gamma] = gamma_prior_r
    # Continuity constraint conversion:
    gamma_tilde_c_prior_r = np.matmul(UT_gamma_p, gamma_tilde_prior_r)

    # Assimilation:
    x_tilde_c_prior_r = np.zeros([N_r + gamma_p * Nc_gamma + J_p])  # Initialising prior state
    x_tilde_c_prior_r[0:N_r] = alpha_prior_r  # Adding alpha prior to prior state
    x_tilde_c_prior_r[N_r: N_r + gamma_p * Nc_gamma] = gamma_tilde_c_prior_r  # Adding gamma prior to prior state


    #######################################################
    # Initialising state and adding prior:
    x_tilde_c_r = np.zeros([N_r + gamma_p * Nc_gamma + J_p, NT])  # Initialising state = [x_0, x_1, ..., x_{NT - 1}]
    x_tilde_c_r[:, 0] = x_tilde_c_prior_r  # Adding prior to states


    #######################################################
    # Initialising evolution operator and additive evolution vector:
    F_r = np.zeros([NT, N_r + gamma_p * Nc_gamma + J_p, N_r + gamma_p * Nc_gamma + J_p])  # Initialising evolution operator F_0, F_1, ..., F_{NT - 1}
    b_r = np.zeros([N_r + gamma_p * Nc_gamma + J_p, NT])  # Initialising additive evolution vector b_0, b_1, ..., b_{NT - 1}


    #######################################################
    # Computing time evolution of reduced model using Crank-Nicolson method:
    t = np.zeros(NT)  # Initialising time array
    for k in range(NT - 1):  # Iterating over time
        J_alpha_star_r, J_gamma_star_r, J_J_star_r = compute_evolution_operator_Jacobians_r(x_tilde_c_r[:, k], t[k])  # Computing evolution operator Jacobian
        F_r[k], b_r[:, k] = compute_evolution_operator_r(x_tilde_c_r[:, k], t[k], J_alpha_star_r, J_gamma_star_r, J_J_star_r)  # Computing evolution operator F and vector b
        x_tilde_c_r[:, k + 1] = np.matmul(F_r[k], x_tilde_c_r[:, k]) + b_r[:, k]  # Time evolution computation
        t[k + 1] = (k + 1) * dt  # Time (hours)


    #######################################################
    # Start of loop for samples of epsilon:
    print('Computing', N_iterations, 'simulations for BAE...')
    for iteration in tqdm(range(N_iterations)):

        #######################################################
        # Drawing sample of parameters for evolution model:

        # Sample of the initial condensation rate I_0(Dp) = I_Dp(Dp, 0):
        I_0 = random.uniform(0.1, 0.5)  # Condensation parameter constant
        I_1 = random.uniform(0, 1.5)  # Condensation parameter inverse quadratic
        def initial_condensation_rate(Dp):
            return I_0 + I_1 / (Dp ** 2)

        # Sample of the deposition rate d(Dp):
        depo_Dpmin = 5  # Deposition parameter; diameter at which minimum
        d_0 = random.uniform(0.3, 0.5)  # Deposition parameter constant
        d_1 = random.uniform(-0.12, 0)  # Deposition parameter linear
        d_2 = -d_1 / (2 * depo_Dpmin)  # Deposition parameter quadratic
        def depo(Dp):
            return d_0 + d_1 * Dp + d_2 * Dp ** 2

        #######################################################
        # Constructing size distribution evolution model:
        F_alpha.add_process('deposition', depo)  # Adding deposition to evolution model
        F_alpha.compile()  # Compiling evolution model

        #######################################################
        # Constructing Jacobian of size distribution evolution model:
        J_F_alpha = GDE_Jacobian(F_alpha)  # Evolution Jacobian
        dF_alpha_d_alpha = J_F_alpha.eval_d_alpha  # Derivative with respect to alpha
        dF_alpha_d_gamma = J_F_alpha.eval_d_gamma  # Derivative with respect to gamma
        dF_alpha_d_J = J_F_alpha.eval_d_J  # Derivative with respect to J

        #######################################################
        # Functions to compute/update evolution operator and Jacobians using Crank-Nicolson method:

        # Function to compute evolution operator Jacobians:
        def compute_evolution_operator_Jacobians(x_tilde_c_star, t_star):
            # Getting current state (i.e. accounting for VAR(p) model):
            x_c_star = np.zeros(N + Nc_gamma + J_p)
            x_c_star[0: N] = x_tilde_c_star[0: N]
            x_c_star[N: N + Nc_gamma] = np.matmul(C_gamma, x_tilde_c_star[N: N + gamma_p * Nc_gamma])
            x_c_star[N + Nc_gamma: N + Nc_gamma + J_p] = x_tilde_c_star[N + gamma_p * Nc_gamma: N + gamma_p * Nc_gamma + J_p]
            # Computing non-constrained state:
            x_star = np.zeros(N + N_gamma + J_p)  # Initialising
            x_star[0: N] = x_c_star[0: N]  # Extracting alpha from constrained state x
            x_star[N: N + N_gamma] = np.matmul(U_gamma, x_c_star[N: N + Nc_gamma])  # Extracting gamma from constrained state x
            x_star[N + N_gamma: N + N_gamma + J_p] = x_c_star[N + Nc_gamma: N + Nc_gamma + J_p]  # Extracting J from constrained state x
            # Computing Jacobians:
            J_alpha_star = dF_alpha_d_alpha(x_star, t_star)
            J_gamma_star = dF_alpha_d_gamma(x_star, t_star)
            J_J_star = dF_alpha_d_J(x_star, t_star)
            return J_alpha_star, J_gamma_star, J_J_star

        # Function to compute evolution operator:
        def compute_evolution_operator(x_tilde_c_star, t_star, J_alpha_star, J_gamma_star, J_J_star):
            # Getting current state (i.e. accounting for VAR(p) model):
            x_c_star = np.zeros(N + Nc_gamma + J_p)
            x_c_star[0: N] = x_tilde_c_star[0: N]
            x_c_star[N: N + Nc_gamma] = np.matmul(C_gamma, x_tilde_c_star[N: N + gamma_p * Nc_gamma])
            x_c_star[N + Nc_gamma: N + Nc_gamma + J_p] = x_tilde_c_star[N + gamma_p * Nc_gamma: N + gamma_p * Nc_gamma + J_p]
            # Computing non-constrained state:
            x_star = np.zeros(N + N_gamma + J_p)  # Initialising
            x_star[0: N] = x_c_star[0: N]  # Extracting alpha from constrained state x
            x_star[N: N + N_gamma] = np.matmul(U_gamma, x_c_star[N: N + Nc_gamma])  # Extracting gamma from constrained state x
            x_star[N + N_gamma: N + N_gamma + J_p] = x_c_star[N + Nc_gamma: N + Nc_gamma + J_p]  # Extracting J from constrained state x
            # Extracting coefficients from state:
            alpha_star = x_star[0: N]
            gamma_star = x_star[N: N + N_gamma]
            J_star = x_star[N + N_gamma: N + N_gamma + 1]
            # Computing F_star:
            F_star = F_alpha.eval(x_star, t_star) - np.matmul(J_alpha_star, alpha_star) - np.matmul(J_gamma_star, gamma_star) - np.matmul(J_J_star, J_star)
            # Computing evolution operators for each coefficient:
            matrix_multiplier = np.linalg.inv(np.eye(N) - (dt / 2) * J_alpha_star)  # Computing matrix multiplier for evolution operators and additive vector
            F_evol_alpha = np.matmul(matrix_multiplier, (np.eye(N) + (dt / 2) * J_alpha_star))
            F_evol_gamma = np.matmul(matrix_multiplier, ((dt / 2) * J_gamma_star))
            F_evol_J = np.matmul(matrix_multiplier, ((dt / 2) * J_J_star))
            # Updating evolution models to account for autoregressive models:
            F_evol_gamma = np.matmul(F_evol_gamma, B_gamma)
            F_evol_J_tilde = np.matmul(F_evol_J, B_J)
            # Updating evolution models to account for continuity constraint:
            F_evol_gamma = np.matmul(F_evol_gamma, U_gamma_p)
            # Computing evolution operator:
            F_evolution = np.zeros([N + gamma_p * Nc_gamma + J_p, N + gamma_p * Nc_gamma + J_p])  # Initialising
            F_evolution[0:N, 0:N] = F_evol_alpha
            F_evolution[0:N, N: N + gamma_p * Nc_gamma] = F_evol_gamma
            F_evolution[0:N, N + gamma_p * Nc_gamma: N + gamma_p * Nc_gamma + J_p] = F_evol_J_tilde
            # Row for gamma:
            F_evolution[N: N + gamma_p * Nc_gamma, N: N + gamma_p * Nc_gamma] = np.matmul(UT_gamma_p, np.matmul(A_gamma, U_gamma_p))
            # Row for J_tilde:
            F_evolution[N + gamma_p * Nc_gamma: N + gamma_p * Nc_gamma + J_p, N + gamma_p * Nc_gamma: N + gamma_p * Nc_gamma + J_p] = A_J
            # Computing evolution additive vector:
            b_evolution = np.zeros(N + gamma_p * Nc_gamma + J_p)  # Initialising
            b_evolution[0:N] = np.matmul(matrix_multiplier, (dt * F_star))
            return F_evolution, b_evolution

        #######################################################
        # Constructing prior:

        # For alpha:
        alpha_prior = F_alpha.compute_coefficients('alpha', initial_guess_size_distribution)  # Computing alpha coefficients from initial condition function

        # For gamma:
        gamma_prior = F_alpha.compute_coefficients('gamma', initial_condensation_rate)  # Computing gamma coefficients from initial guess of condensation rate
        # Prior for gamma_tilde:
        gamma_tilde_prior = np.zeros([gamma_p * N_gamma])
        for i in range(gamma_p):
            gamma_tilde_prior[i * N_gamma: (i + 1) * N_gamma] = gamma_prior
        # Continuity constraint conversion:
        gamma_tilde_c_prior = np.matmul(UT_gamma_p, gamma_tilde_prior)

        # Assimilation:
        x_tilde_c_prior = np.zeros([N + gamma_p * Nc_gamma + J_p])  # Initialising prior state
        x_tilde_c_prior[0:N] = alpha_prior  # Adding alpha prior to prior state
        x_tilde_c_prior[N: N + gamma_p * Nc_gamma] = gamma_tilde_c_prior  # Adding gamma prior to prior state

        #######################################################
        # Initialising state and adding prior:
        x_tilde_c = np.zeros([N + gamma_p * Nc_gamma + J_p, NT])  # Initialising state = [x_0, x_1, ..., x_{NT - 1}]
        x_tilde_c[:, 0] = x_tilde_c_prior  # Adding prior to states

        #######################################################
        # Initialising evolution operator and additive evolution vector:
        F = np.zeros([NT, N + gamma_p * Nc_gamma + J_p, N + gamma_p * Nc_gamma + J_p])  # Initialising evolution operator F_0, F_1, ..., F_{NT - 1}
        b = np.zeros([N + gamma_p * Nc_gamma + J_p, NT])  # Initialising additive evolution vector b_0, b_1, ..., b_{NT - 1}

        #######################################################
        # Computing time evolution and approximation error for each time step:
        for k in range(NT - 1):  # Iterating over time
            J_alpha_star, J_gamma_star, J_J_star = compute_evolution_operator_Jacobians(x_tilde_c[:, k], t[k])  # Computing evolution operator Jacobian
            F[k], b[:, k] = compute_evolution_operator(x_tilde_c[:, k], t[k], J_alpha_star, J_gamma_star, J_J_star)  # Computing evolution operator F and vector b
            compute_epsilon_matrix = np.matmul(P, F[k]) - np.matmul(F_r[k], P)  # Approximation error computation matrix
            compute_epsilon_vector = np.matmul(P, b[:, k]) - b_r[:, k]  # Approximation error computation vector
            epsilon[iteration, :, k] = np.matmul(compute_epsilon_matrix, x_tilde_c[:, k]) + compute_epsilon_vector  # Computing approximation error
            x_tilde_c[:, k + 1] = np.matmul(F[k], x_tilde_c[:, k]) + b[:, k]  # Time evolution computation
        # Computing approximation error at final time:
        compute_epsilon_matrix = np.matmul(P, F[NT - 1]) - np.matmul(F_r[NT - 1], P)  # Approximation error computation matrix
        compute_epsilon_vector = np.matmul(P, b[:, NT - 1]) - b_r[:, NT - 1]  # Approximation error computation vector
        epsilon[iteration, :, NT - 1] = np.matmul(compute_epsilon_matrix, x_tilde_c[:, NT - 1]) + compute_epsilon_vector  # Computing approximation error


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

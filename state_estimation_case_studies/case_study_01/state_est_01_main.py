"""

Title: Computes estimates of the particle size distribution, condensation rate, deposition rate, and nucleation rate
Author: Vincent Russell
Date: June 18, 2021

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
from basic_tools import Extended_Kalman_filter, compute_fixed_interval_extended_Kalman_smoother
from observation_models.data.simulated import load_observations
from evolution_models.tools import GDE_evolution_model, change_basis_volume_to_diameter_sorc, GDE_Jacobian, compute_U
from observation_models.tools import Size_distribution_observation_model, Size_distribution_observation_model_Jacobian


#######################################################
# Importing parameter file:
from state_estimation_case_studies.case_study_01.state_est_01_parameters import *


#######################################################
if __name__ == '__main__':

    #######################################################
    # Initialising timer for total computation:
    initial_time = tm.time()  # Initial time stamp


    #######################################################
    # Importing simulated observations and true size distribution:
    observation_data = load_observations('simulated_observations')  # Loading data file
    d_obs, Y = observation_data['d_obs'], observation_data['Y']  # Extracting observations
    # d_true, n_v_true = observation_data['d_true'], observation_data['n_true']  # Extracting true size distribution
    # v_obs = basic_tools.diameter_to_volume(d_obs)  # Converting diameter observations to volume
    # v_true = basic_tools.diameter_to_volume(d_true)  # Converting true diameter to volume


    #######################################################
    # Constructing size distribution evolution model:
    F_alpha = GDE_evolution_model(Ne, Np, vmin, vmax, dt, NT, boundary_zero=boundary_zero)  # Initialising evolution model
    F_alpha.add_unknown('condensation', Ne_gamma, Np_gamma)  # Adding condensation as unknown to evolution model
    F_alpha.add_unknown('deposition', Ne_eta, Np_eta)  # Adding deposition as unknown to evolution model
    F_alpha.add_unknown('source')  # Adding source as unknown to evolution model
    F_alpha.add_process('coagulation', coag, load_coagulation=load_coagulation)  # Adding coagulation to evolution model
    F_alpha.compile(time_integrator='euler')  # Compiling evolution model and adding time integrator


    #######################################################
    # Constructing condensation rate evolution model:
    def F_gamma(gamma):
        return gamma


    #######################################################
    # Constructing deposition rate evolution model:
    def F_eta(eta):
        return eta


    #######################################################
    # Constructing source (nucleation) rate evolution model:
    A_J = np.zeros([J_p, J_p])  # Initialising AR(p) matrix
    for i in range(J_p):
        A_J[0, i] = J_a[i]  # Computing elements in AR(p) matrix
    A_J = A_J + np.eye(J_p, J_p, -1)  # Adding ones to off-diagonal
    def F_J_tilde(J_tilde):  # J_tilde_k = [J_k, J_{k - 1}]
        return np.matmul(A_J, J_tilde)


    #######################################################
    # Continuity constraint computations for gamma (condensation):
    num_constraints_gamma = Ne_gamma - 1  # Number of contraints
    Nc_gamma = N_gamma - num_constraints_gamma  # Dimensions of constrained gamma
    U_gamma, UT_gamma = compute_U(N_gamma, Ne_gamma, Np_gamma, F_alpha.phi_gamma, F_alpha.x_boundaries_gamma)  # Computing null space continuity matrix


    #######################################################
    # Continuity constraint computations for eta (deposition):
    num_constraints_eta = Ne_eta - 1  # Number of contraints
    Nc_eta = N_eta - num_constraints_eta  # Dimensions of constrained eta
    U_eta, UT_eta = compute_U(N_eta, Ne_eta, Np_eta, F_alpha.phi_eta, F_alpha.x_boundaries_eta)  # Computing null space continuity matrix


    #######################################################
    # Constructing state evolution model:
    def F(x_c, t):
        output = np.zeros(N + Nc_gamma + Nc_eta + J_p)  # Initialising output
        # Computing non-constrained state:
        x = np.zeros(N + N_gamma + N_eta + J_p)  # Initialising
        x[0: N] = x_c[0: N]  # Extracting alpha from constrained state x
        x[N: N + N_gamma] = np.matmul(U_gamma, x_c[N: N + Nc_gamma])  # Extracting gamma from constrained state x
        x[N + N_gamma: N + N_gamma + N_eta] = np.matmul(U_eta, x_c[N + Nc_gamma: N + Nc_gamma + Nc_eta])  # Extracting eta from constrained state x
        x[N + N_gamma + N_eta: N + N_gamma + N_eta + J_p] = x_c[N + Nc_gamma + Nc_eta: N + Nc_gamma + Nc_eta + J_p]  # Extracting J from constrained state x
        # Extracting coefficients from state:
        gamma = x[N: N + N_gamma]
        eta = x[N + N_gamma: N + N_gamma + N_eta]
        J_tilde = x[N + N_gamma + N_eta: N + N_gamma + N_eta + J_p]
        # Computing output:
        output[0: N] = F_alpha.eval(x, t)
        output[N: N + Nc_gamma] = np.matmul(UT_gamma, F_gamma(gamma))
        output[N + Nc_gamma: N + Nc_gamma + Nc_eta] = np.matmul(UT_eta, F_eta(eta))
        output[N + Nc_gamma + Nc_eta: N + Nc_gamma + Nc_eta + J_p] = F_J_tilde(J_tilde)
        return output


    #######################################################
    # Constructing Jacobian of size distribution evolution model:
    J_F_alpha = GDE_Jacobian(F_alpha)  # Evolution Jacobian
    dF_alpha_d_alpha = J_F_alpha.eval_d_alpha  # Derivative with respect to alpha
    dF_alpha_d_gamma = J_F_alpha.eval_d_gamma  # Derivative with respect to gamma
    dF_alpha_d_eta = J_F_alpha.eval_d_eta  # Derivative with respect to eta
    dF_alpha_d_J = J_F_alpha.eval_d_J  # Derivative with respect to J
    # Derivative with respect to J_tilde:
    def dF_alpha_d_J_tilde():
        output = np.zeros([N, J_p])
        output[:, 0] = dF_alpha_d_J()[:, 0]
        return output


    #######################################################
    # Constructing Jacobian of condensation rate evolution model:
    def dF_gamma_d_alpha():
        return np.zeros([N_gamma, N])
    def dF_gamma_d_gamma():
        return np.eye(N_gamma)
    def dF_gamma_d_eta():
        return np.zeros([N_gamma, N_eta])
    def dF_gamma_d_J_tilde():
        return np.zeros([N_gamma, J_p])


    #######################################################
    # Constructing Jacobian of deposition rate evolution model:
    def dF_eta_d_alpha():
        return np.zeros([N_eta, N])
    def dF_eta_d_gamma():
        return np.zeros([N_eta, N_gamma])
    def dF_eta_d_eta():
        return np.eye(N_eta)
    def dF_eta_d_J_tilde():
        return np.zeros([N_eta, J_p])


    #######################################################
    # Constructing Jacobian of nucleation rate evolution model:
    def dF_J_tilde_d_alpha():
        return np.zeros([J_p, N])
    def dF_J_tilde_d_gamma():
        return np.zeros([J_p, N_gamma])
    def dF_J_tilde_d_eta():
        return np.zeros([J_p, N_eta])
    def dF_J_tilde_d_J_tilde():
        return A_J


    #######################################################
    # Constructing Jacobian of state evolution model:
    def J_F(x_c, t):
        output = np.zeros([N + Nc_gamma + Nc_eta + J_p, N + Nc_gamma + Nc_eta + J_p])  # Initialising output
        # Computing non-constrained state:
        x = np.zeros(N + N_gamma + N_eta + J_p)  # Initialising
        x[0: N] = x_c[0: N]  # Extracting alpha from constrained state x
        x[N: N + N_gamma] = np.matmul(U_gamma, x_c[N: N + Nc_gamma])  # Extracting gamma from constrained state x
        x[N + N_gamma: N + N_gamma + N_eta] = np.matmul(U_eta, x_c[N + Nc_gamma: N + Nc_gamma + Nc_eta])  # Extracting eta from constrained state x
        x[N + N_gamma + N_eta: N + N_gamma + N_eta + J_p] = x_c[N + Nc_gamma + Nc_eta: N + Nc_gamma + Nc_eta + J_p]  # Extracting J from constrained state x
        # Extracting alpha from state:
        alpha = x[0: N]
        # Computing output:
        # Derivatives of F_alpha:
        output[0:N, 0:N] = dF_alpha_d_alpha(x, t)
        output[0:N, N: N + Nc_gamma] = np.matmul(dF_alpha_d_gamma(alpha), U_gamma)
        output[0:N, N + Nc_gamma: N + Nc_gamma + Nc_eta] = np.matmul(dF_alpha_d_eta(alpha), U_eta)
        output[0:N, N + Nc_gamma + Nc_eta: N + Nc_gamma + Nc_eta + J_p] = dF_alpha_d_J_tilde()
        # Derivatives of F_gamma:
        output[N: N + Nc_gamma, 0:N] = np.matmul(UT_gamma, dF_gamma_d_alpha())
        output[N: N + Nc_gamma, N: N + Nc_gamma] = np.matmul(UT_gamma, np.matmul(dF_gamma_d_gamma(), U_gamma))
        output[N: N + Nc_gamma, N + Nc_gamma: N + Nc_gamma + Nc_eta] = np.matmul(UT_gamma, np.matmul(dF_gamma_d_eta(), U_eta))
        output[N: N + Nc_gamma, N + Nc_gamma + Nc_eta: N + Nc_gamma + Nc_eta + J_p] = np.matmul(UT_gamma, dF_gamma_d_J_tilde())
        # Derivatives of F_eta:
        output[N + Nc_gamma: N + Nc_gamma + Nc_eta, 0:N] = np.matmul(UT_eta, dF_eta_d_alpha())
        output[N + Nc_gamma: N + Nc_gamma + Nc_eta, N: N + Nc_gamma] = np.matmul(UT_eta, np.matmul(dF_eta_d_gamma(), U_gamma))
        output[N + Nc_gamma: N + Nc_gamma + Nc_eta, N + Nc_gamma: N + Nc_gamma + Nc_eta] = np.matmul(UT_eta, np.matmul(dF_eta_d_eta(), U_eta))
        output[N + Nc_gamma: N + Nc_gamma + Nc_eta, N + Nc_gamma + Nc_eta: N + Nc_gamma + Nc_eta + J_p] = np.matmul(UT_eta, dF_eta_d_J_tilde())
        # Derivatives of F_J:
        output[N + Nc_gamma + Nc_eta: N + Nc_gamma + Nc_eta + J_p, 0:N] = dF_J_tilde_d_alpha()
        output[N + Nc_gamma + Nc_eta: N + Nc_gamma + Nc_eta + J_p, N: N + Nc_gamma] = np.matmul(dF_J_tilde_d_gamma(), U_gamma)
        output[N + Nc_gamma + Nc_eta: N + Nc_gamma + Nc_eta + J_p, N + Nc_gamma: N + Nc_gamma + Nc_eta] = np.matmul(dF_J_tilde_d_eta(), U_eta)
        output[N + Nc_gamma + Nc_eta: N + Nc_gamma + Nc_eta + J_p, N + Nc_gamma + Nc_eta: N + Nc_gamma + Nc_eta + J_p] = dF_J_tilde_d_J_tilde()
        return output


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
        output = np.zeros([M, N + Nc_gamma + Nc_eta + J_p])
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
    Gamma_alpha_w[0: Np, 0: Np] = alpha_first_element_multiplier * Gamma_alpha_w[0: Np, 0: Np]  # First element multiplier

    # For gamma:
    sigma_gamma_w = np.array([sigma_gamma_w_0, sigma_gamma_w_1, sigma_gamma_w_2, sigma_gamma_w_3, sigma_gamma_w_4, sigma_gamma_w_5, sigma_gamma_w_6])  # Array of standard deviations
    Gamma_gamma_w = basic_tools.compute_correlated_covariance_matrix(N_gamma, Np_gamma, Ne_gamma, sigma_gamma_w, sigma_gamma_w_correlation)  # Covariance matrix computation
    Gamma_gamma_w[0: Np_gamma, 0: Np_gamma] = gamma_first_element_multiplier * Gamma_gamma_w[0: Np_gamma, 0: Np_gamma]  # First element multiplier
    Gamma_gamma_c_w = np.matmul(UT_gamma, np.matmul(Gamma_gamma_w, U_gamma))  # Continuity constraint conversion

    # For eta:
    sigma_eta_w = np.array([sigma_eta_w_0, sigma_eta_w_1, sigma_eta_w_2, sigma_eta_w_3, sigma_eta_w_4, sigma_eta_w_5, sigma_eta_w_6])  # Array of standard deviations
    Gamma_eta_w = basic_tools.compute_correlated_covariance_matrix(N_eta, Np_eta, Ne_eta, sigma_eta_w, sigma_eta_w_correlation)  # Covariance matrix computation
    Gamma_eta_w[0: Np_eta, 0: Np_eta] = eta_first_element_multiplier * Gamma_eta_w[0: Np_eta, 0: Np_eta]  # First element multiplier
    Gamma_eta_c_w = np.matmul(UT_eta, np.matmul(Gamma_eta_w, U_eta))  # Continuity constraint conversion

    # For J_tilde:
    Gamma_J_tilde_w = np.zeros([J_p, J_p])
    Gamma_J_tilde_w[0, 0] = sigma_J_w ** 2

    # Assimilation:
    Gamma_w = np.zeros([N + Nc_gamma + Nc_eta + J_p, N + Nc_gamma + Nc_eta + J_p])  # Initialising noise covariance for state
    Gamma_w[0:N, 0:N] = Gamma_alpha_w  # Adding alpha covariance to state covariance
    Gamma_w[N: N + Nc_gamma, N: N + Nc_gamma] = Gamma_gamma_c_w  # Adding gamma covariance to state covariance
    Gamma_w[N + Nc_gamma: N + Nc_gamma + Nc_eta, N + Nc_gamma: N + Nc_gamma + Nc_eta] = Gamma_eta_c_w  # Adding eta covariance to state covariance
    Gamma_w[N + Nc_gamma + Nc_eta: N + Nc_gamma + Nc_eta + J_p, N + Nc_gamma + Nc_eta: N + Nc_gamma + Nc_eta + J_p] = Gamma_J_tilde_w  # Adding J_tilde covariance to state covariance


    #######################################################
    # Constructing prior and prior covariance:

    # For alpha:
    alpha_prior = F_alpha.compute_coefficients('alpha', initial_guess_size_distribution)  # Computing alpha coefficients from initial condition function
    Gamma_alpha_prior = np.eye(N)
    for i in range(N):
        if i % Np == 0:
            Gamma_alpha_prior[i, i] = (sigma_alpha_prior_0 ** 2)
        elif i % Np == 1:
            Gamma_alpha_prior[i, i] = (sigma_alpha_prior_1 ** 2)
        elif i % Np == 2:
            Gamma_alpha_prior[i, i] = (sigma_alpha_prior_2 ** 2)
        elif i % Np == 3:
            Gamma_alpha_prior[i, i] = (sigma_alpha_prior_3 ** 2)
        elif i % Np == 4:
            Gamma_alpha_prior[i, i] = (sigma_alpha_prior_4 ** 2)
        elif i % Np == 5:
            Gamma_alpha_prior[i, i] = (sigma_alpha_prior_5 ** 2)
        elif i % Np == 6:
            Gamma_alpha_prior[i, i] = (sigma_alpha_prior_6 ** 2)
    # First element multiplier:
    Gamma_alpha_prior[0: Np, 0: Np] = alpha_first_element_multiplier * Gamma_alpha_prior[0: Np, 0: Np]

    # For gamma:
    gamma_prior = F_alpha.compute_coefficients('gamma', initial_guess_condensation_rate)    # Computing gamma coefficients from initial guess of condensation rate
    Gamma_gamma_prior = np.eye(N_gamma)
    for i in range(N_gamma):
        if i % Np_gamma == 0:
            Gamma_gamma_prior[i, i] = (sigma_gamma_prior_0 ** 2)
        elif i % Np_gamma == 1:
            Gamma_gamma_prior[i, i] = (sigma_gamma_prior_1 ** 2)
        elif i % Np_gamma == 2:
            Gamma_gamma_prior[i, i] = (sigma_gamma_prior_2 ** 2)
        elif i % Np_gamma == 3:
            Gamma_gamma_prior[i, i] = (sigma_gamma_prior_3 ** 2)
        elif i % Np_gamma == 4:
            Gamma_gamma_prior[i, i] = (sigma_gamma_prior_4 ** 2)
        elif i % Np_gamma == 5:
            Gamma_gamma_prior[i, i] = (sigma_gamma_prior_5 ** 2)
        elif i % Np_gamma == 6:
            Gamma_gamma_prior[i, i] = (sigma_gamma_prior_6 ** 2)
    # First element multiplier:
    Gamma_gamma_prior[0: Np_gamma, 0: Np_gamma] = gamma_first_element_multiplier * Gamma_gamma_prior[0: Np_gamma, 0: Np_gamma]
    # Continuity constraint conversion:
    gamma_c_prior = np.matmul(UT_gamma, gamma_prior)
    Gamma_gamma_c_prior = np.matmul(UT_gamma, np.matmul(Gamma_gamma_prior, U_gamma))

    # For eta:
    eta_prior = F_alpha.compute_coefficients('eta', initial_guess_deposition_rate)    # Computing eta coefficients from initial guess of deposition rate
    Gamma_eta_prior = np.eye(N_eta)
    for i in range(N_eta):
        if i % Np_eta == 0:
            Gamma_eta_prior[i, i] = (sigma_eta_prior_0 ** 2)
        elif i % Np_eta == 1:
            Gamma_eta_prior[i, i] = (sigma_eta_prior_1 ** 2)
        elif i % Np_eta == 2:
            Gamma_eta_prior[i, i] = (sigma_eta_prior_2 ** 2)
        elif i % Np_eta == 3:
            Gamma_eta_prior[i, i] = (sigma_eta_prior_3 ** 2)
        elif i % Np_eta == 4:
            Gamma_eta_prior[i, i] = (sigma_eta_prior_4 ** 2)
        elif i % Np_eta == 5:
            Gamma_eta_prior[i, i] = (sigma_eta_prior_5 ** 2)
        elif i % Np_eta == 6:
            Gamma_eta_prior[i, i] = (sigma_eta_prior_6 ** 2)
    # First element multiplier:
    Gamma_eta_prior[0: Np_eta, 0: Np_eta] = eta_first_element_multiplier * Gamma_eta_prior[0: Np_eta, 0: Np_eta]
    # Continuity constraint conversion:
    eta_c_prior = np.matmul(UT_eta, eta_prior)
    Gamma_eta_c_prior = np.matmul(UT_eta, np.matmul(Gamma_eta_prior, U_eta))

    # For J_tilde:
    Gamma_J_tilde_prior = np.zeros([J_p, J_p])
    for i in range(J_p):
        Gamma_J_tilde_prior[i, i] = sigma_J_prior ** 2

    # Assimilation:
    x_prior = np.zeros([N + Nc_gamma + Nc_eta + J_p])  # Initialising prior state
    x_prior[0:N] = alpha_prior  # Adding alpha prior to prior state
    x_prior[N: N + Nc_gamma] = gamma_c_prior  # Adding gamma prior to prior state
    x_prior[N + Nc_gamma: N + Nc_gamma + Nc_eta] = eta_c_prior  # Adding eta prior to prior state
    Gamma_prior = np.zeros([N + Nc_gamma + Nc_eta + J_p, N + Nc_gamma + Nc_eta + J_p])  # Initialising prior covariance for state
    Gamma_prior[0:N, 0:N] = Gamma_alpha_prior  # Adding alpha covariance to state covariance
    Gamma_prior[N: N + Nc_gamma, N: N + Nc_gamma] = Gamma_gamma_c_prior  # Adding gamma covariance to state covariance
    Gamma_prior[N + Nc_gamma: N + Nc_gamma + Nc_eta, N + Nc_gamma: N + Nc_gamma + Nc_eta] = Gamma_eta_c_prior  # Adding eta covariance to state covariance
    Gamma_prior[N + Nc_gamma + Nc_eta: N + Nc_gamma + Nc_eta + J_p, N + Nc_gamma + Nc_eta: N + Nc_gamma + Nc_eta + J_p] = Gamma_J_tilde_prior  # Adding J_tilde covariance to state covariance


    #######################################################
    # Initialising state and adding prior:
    x_c = np.zeros([N + Nc_gamma + Nc_eta + J_p, NT])  # Initialising state = [x_0, x_1, ..., x_{NT - 1}]
    Gamma_c = np.zeros([NT, N + Nc_gamma + Nc_eta + J_p, N + Nc_gamma + Nc_eta + J_p])  # Initialising state covariance matrix Gamma_0, Gamma_1, ..., Gamma_NT
    x_c_predict = np.zeros([N + Nc_gamma + Nc_eta + J_p, NT])  # Initialising predicted state
    Gamma_c_predict = np.zeros([NT, N + Nc_gamma + Nc_eta + J_p, N + Nc_gamma + Nc_eta + J_p])  # Initialising predicted state covariance
    x_c[:, 0], x_c_predict[:, 0] = x_prior, x_prior  # Adding prior to states
    Gamma_c[0], Gamma_c_predict[0] = Gamma_prior, Gamma_prior  # Adding prior to state covariance matrices


    #######################################################
    # Constructing extended Kalman filter model:
    model = Extended_Kalman_filter(F, J_F, H, J_H, Gamma_w, Gamma_v, NT)


    #######################################################
    # Computing time evolution of model:
    print('Computing Extended Kalman filter estimates...')
    t = np.zeros(NT)  # Initialising time array
    for k in tqdm(range(NT - 1)):  # Iterating over time
        x_c_predict[:, k + 1], Gamma_c_predict[k + 1] = model.predict(x_c[:, k], Gamma_c[k], t[k], k)  # Computing prediction
        x_c[:, k + 1], Gamma_c[k + 1],  = model.update(x_c_predict[:, k + 1], Gamma_c_predict[k + 1], Y[:, k + 1], t[k], k)  # Computing update
        t[k + 1] = (k + 1) * dt  # Time (hours)


    #######################################################
    # Computing smoothed estimates:
    if smoothing:
        print('Computing Extended Kalman smoother estimates...')
        x_c, Gamma_c = compute_fixed_interval_extended_Kalman_smoother(J_F, dt, NT, N + Nc_gamma + Nc_eta + J_p, x_c, Gamma_c, x_c_predict, Gamma_c_predict)


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
    d_plot, v_plot, n_Dp_plot, sigma_n_Dp = F_alpha.get_nplot_discretisation(alpha, Gamma_alpha=Gamma_alpha, convert_v_to_Dp=True)
    n_Dp_plot_upper = n_Dp_plot + 2 * sigma_n_Dp
    n_Dp_plot_lower = n_Dp_plot - 2 * sigma_n_Dp
    # Condensation rate:
    _, d_plot_cond, cond_Dp_plot, sigma_cond_Dp = F_alpha.get_parameter_estimation_discretisation('condensation', gamma, Gamma_gamma)
    cond_Dp_plot_upper = cond_Dp_plot + 2 * sigma_cond_Dp
    cond_Dp_plot_lower = cond_Dp_plot - 2 * sigma_cond_Dp
    # Deposition rate:
    _, d_plot_depo, depo_plot, sigma_depo = F_alpha.get_parameter_estimation_discretisation('deposition', eta, Gamma_eta)
    depo_plot_upper = depo_plot + 2 * sigma_depo
    depo_plot_lower = depo_plot - 2 * sigma_depo
    # Nucleation rate:
    J_Dp_plot, sigma_J_Dp = F_alpha.get_parameter_estimation_discretisation('nucleation', J, Gamma_J, convert_v_to_Dp=True)
    J_Dp_plot_upper = J_Dp_plot + 2 * sigma_J_Dp
    J_Dp_plot_lower = J_Dp_plot - 2 * sigma_J_Dp


    #######################################################
    # Computing true underlying parameters plotting discretisation:
    Nplot_cond = len(d_plot_cond)  # Length of size discretisation
    Nplot_depo = len(d_plot_depo)  # Length of size discretisation
    cond_Dp_true_plot = np.zeros([Nplot_cond, NT])  # Initialising volume-based condensation rate
    depo_true_plot = np.zeros([Nplot_depo, NT])  # Initialising deposition rate
    sorc_v_true_plot = np.zeros(NT)  # Initialising volume-based source (nucleation) rate
    for k in range(NT):
        sorc_v_true_plot[k] = sorc(t[k])  # Computing volume-based nucleation rate
        for i in range(Nplot_cond):
            cond_Dp_true_plot[i, k] = cond(d_plot_cond[i])  # Computing volume-based condensation rate
        for i in range(Nplot_depo):
            depo_true_plot[i, k] = depo(d_plot_depo[i])  # Computing deposition rate
    sorc_Dp_true_plot = change_basis_volume_to_diameter_sorc(sorc_v_true_plot, Dp_min)  # Computing diameter-based nucleation rate


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
    ylimits = [0, 12000]  # Plot boundary limits for y-axis
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
    legend_cond = ['Estimate', '$\pm 2 \sigma$', '', 'Truth']  # Adding legend to plot
    line_color_cond = ['blue', 'blue', 'blue', 'green']  # Colors of lines in plot
    line_style_cond = ['solid', 'dashed', 'dashed', 'solid']  # Style of lines in plot
    delay_cond = 0  # Delay for each frame (ms)

    # Parameters for deposition plot:
    ylimits_depo = [0, 1]  # Plot boundary limits for y-axis
    xlabel_depo = '$D_p$ ($\mu$m)'  # x-axis label for plot
    ylabel_depo = '$d(D_p)$ (hour$^{-1}$)'  # y-axis label for plot
    title_depo = 'Deposition rate estimation'  # Title for plot
    location_depo = location + '3'  # Location for plot
    legend_depo = ['Estimate', '$\pm 2 \sigma$', '', 'Truth']  # Adding legend to plot
    line_color_depo = ['blue', 'blue', 'blue', 'green']  # Colors of lines in plot
    line_style_depo = ['solid', 'dashed', 'dashed', 'solid']  # Style of lines in plot
    delay_depo = delay_cond  # Delay between frames in milliseconds

    # Size distribution animation:
    basic_tools.plot_1D_animation(d_plot, n_Dp_plot, n_Dp_plot_lower, n_Dp_plot_upper, plot_add=(d_obs, Y), xlimits=xlimits, ylimits=ylimits, xscale=xscale, xlabel=xlabel, ylabel=ylabel, title=title,
                                  location=location, legend=legend, time=time, timetext=timetext, line_color=line_color, line_style=line_style, doing_mainloop=False)

    # Condensation rate animation:
    basic_tools.plot_1D_animation(d_plot_cond, cond_Dp_plot, cond_Dp_plot_lower, cond_Dp_plot_upper, cond_Dp_true_plot, xlimits=xlimits, ylimits=ylimits_cond, xscale=xscale, xlabel=xlabel_cond, ylabel=ylabel_cond, title=title_cond,
                                  location=location_cond, delay=delay_cond, legend=legend_cond, time=time, timetext=timetext, line_color=line_color_cond, line_style=line_style_cond, doing_mainloop=False)

    # Deposition rate animation:
    basic_tools.plot_1D_animation(d_plot_depo, depo_plot, depo_plot_lower, depo_plot_upper, depo_true_plot, xlimits=xlimits, ylimits=ylimits_depo, xscale=xscale, xlabel=xlabel_depo, ylabel=ylabel_depo, title=title_depo,
                                  location=location_depo, delay=delay_depo, legend=legend_depo, time=time, timetext=timetext, line_color=line_color_depo, line_style=line_style_depo, doing_mainloop=False)

    # Mainloop and print:
    if plot_animations:
        print('Plotting animations...')
        mainloop()  # Runs tkinter GUI for plots and animations


    #######################################################
    # Plotting nucleation rate:
    if plot_nucleation:
        print('Plotting nucleation...')
        figJ, axJ = plt.subplots(figsize=(8.00, 5.00), dpi=100)
        plt.plot(time, J_Dp_plot, 'b-', label='Estimate')
        plt.plot(time, J_Dp_plot_lower, 'b--', label='$\pm 2 \sigma$')
        plt.plot(time, J_Dp_plot_upper, 'b--')
        plt.plot(time, sorc_Dp_true_plot, 'g-', label='Truth')
        axJ.set_xlim([0, T])
        axJ.set_ylim([0, 3500])
        axJ.set_xlabel('$t$ (hour)', fontsize=12)
        axJ.set_ylabel('$J(t)$ \n $(\mu$m$^{-1}$cm$^{-3}$ hour$^{-1}$)', fontsize=12, rotation=0)
        axJ.yaxis.set_label_coords(-0.015, 1.02)
        axJ.set_title('Nucleation rate esimation', fontsize=12)
        axJ.grid()
        axJ.legend(fontsize=11, loc='upper left')


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
    image_max = 12000  # Maximum of image colour
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

"""
General dynamic equation evolution operator
"""


#######################################################
# Modules:
import numpy as np

# Local modules:
from basic_tools import print_lines, get_kwarg_value, Zero, volume_to_diameter
from evolution_models.data import load_coagulation_tensors
from evolution_models.tools import (get_Legendre_basis, get_Legendre_basis_derivative,
                                    get_discretisation, get_plotting_discretisation, compute_coefficients,
                                    compute_M, compute_A, compute_Q, compute_D, compute_B_C,
                                    compute_Q_gamma, compute_D_eta,
                                    compute_R, compute_R1_gamma, compute_R2_gamma,
                                    Condensation_evolution, Condensation_unknown_evolution,
                                    Source_evolution, Source_unknown_evolution,
                                    Deposition_evolution, Deposition_unknown_evolution,
                                    Coagulation_evolution,
                                    get_f, get_next_step, change_basis_operator, get_element_matrix)


#######################################################
# General dynamic equation evolution model class:
class GDE_evolution_model:

    def __init__(self, Ne, Np, xmin, xmax, dt, NT, **kwargs):
        # Print statement:
        print_lines()
        print('Initialising general dynamic equation evolution operator.')
        # Setup parameters:
        self.scale_type = get_kwarg_value(kwargs, 'scale_type', 'linear')  # Size discretisation linear or log formulation ('linear' or 'log')
        self.boundary_zero = get_kwarg_value(kwargs, 'boundary_zero', True)  # Set to True for imposing boundary condition n(vmin, t) = 0
        # Time parameters:
        self.dt = dt  # Time step
        self.NT = NT  # Total number of time steps
        # Domain parameters:
        self.xlim = [xmin, xmax]  # Domain limits
        if self.scale_type == 'log':
            self.Dp_lim = volume_to_diameter(np.exp(self.xlim))  # Domain limits in particle diameter
        else:
            self.Dp_lim = volume_to_diameter(self.xlim)  # Domain limits in particle diameter
        self.logDp_lim = np.log(self.Dp_lim)
        # Discretisation parameters:
        self.Ne = Ne  # Number of elements
        self.Np = Np  # Np - 1 = degree of Legendre polynomial approximation in each element
        self.N = Ne * Np  # Total degrees of freedom
        # Uniform discretisation; x_boundaries into Ne elements, with element size h, and Gauss nodes in each element:
        self.x_Gauss, self.x_boundaries, self.h = get_discretisation(self.Ne, self.Np, self.xlim[0], self.xlim[1])
        # Basis functions:
        self.phi = get_Legendre_basis(self.N, self.Np, self.x_boundaries)  # Basis function; phi[j](x) = phi_j(x) for j = 0, 1, ..., N - 1
        self.dphi = get_Legendre_basis_derivative(self.N, self.Np, self.x_boundaries, self.phi)  # Derivative of basis function; dphi[j](x) = phi_j'(x) for j = 0, 1, ..., N - 1
        # Model-independent matrix computations:
        self.M = compute_M(self.N, self.Np, self.h)
        self.A = compute_A(self.N, self.x_Gauss, self.phi)
        # Matrix M_ell^-1 computation:
        self.inv_M = np.zeros([self.Ne, self.Np, self.Np])  # Initialising set of elemental matrices M_ell^-1
        for ell in range(self.Ne):  # Iterating over elements
            self.inv_M[ell] = np.linalg.inv(get_element_matrix(self.M, ell, self.Np))  # Getting M_ell^-1 matrix
        # Vectors, matrices, and tensors initialisation:
        self.Q, self.Q_gamma = None, None
        self.R, self.R1_gamma, self.R2_gamma = None, None, None
        self.D, self.D_eta = None, None
        self.B, self.C = None, None
        # Process models initialisation:
        self.cond = Zero.function
        self.sorc = Zero.function
        self.depo = Zero.function
        self.coag = Zero.function
        # Time evolution operators initialisation:
        self.f_cond = Zero.function
        self.f_sorc = Zero.function
        self.f_depo = Zero.function
        self.f_coag = Zero.function
        # Compile functions initialisation:
        self.f = Zero.function
        self.time_integrator = Zero.function
        self.next_step = Zero.function
        # Unknown processes initialisation:
        self.unknowns = list()
        self.Ne_gamma, self.Np_gamma, self.N_gamma = None, None, None
        self.phi_gamma, self.x_boundaries_gamma, self.h_gamma = None, None, None
        self.Ne_eta, self.Np_eta, self.N_eta = None, None, None
        self.phi_eta, self.x_boundaries_eta, self.h_eta = None, None, None

    # Function to add process models to evolution model:
    def add_process(self, process, model, **kwargs):
        print('Adding', process, 'to model...')
        if process == 'condensation':
            self.cond = model
            self.Q = compute_Q(self.cond, self.N, self.Np, self.x_boundaries, self.phi, self.dphi, self.scale_type)
            self.R = compute_R(self.cond, self.Ne, self.Np, self.N, self.x_boundaries, self.phi, self.inv_M, self.Q, self.boundary_zero, self.scale_type)
            self.f_cond = Condensation_evolution(self.R).eval
        elif process == 'source':
            self.sorc = model
            self.f_sorc = Source_evolution(self, self.sorc).eval
        elif process == 'deposition':
            self.depo = model
            self.D = compute_D(self.depo, self.N, self.Np, self.x_boundaries, self.phi, self.scale_type)
            self.f_depo = Deposition_evolution(self).eval
        elif process == 'coagulation':
            self.coag = model
            load_coagulation = get_kwarg_value(kwargs, 'load_coagulation', False)  # Set to True to load coagulation tensors
            save_coagulation = get_kwarg_value(kwargs, 'save_coagulation', False)  # Set to True to save coagulation tensors
            coagulation_suffix = get_kwarg_value(kwargs, 'coagulation_suffix', False)  # Set to True to save coagulation tensors
            # Loading or computing coagulation tensors:
            if load_coagulation:
                print('Loading coagulation data...')
                if not coagulation_suffix:
                    data_filename = 'Coagulation_data_Ne=' + str(self.Ne) + '_Np=' + str(self.Np) + '_' + self.scale_type
                else:
                    data_filename = 'Coagulation_data_Ne=' + str(self.Ne) + '_Np=' + str(self.Np) + '_' + self.scale_type + '_' + coagulation_suffix
                coagulation_data = load_coagulation_tensors(data_filename)
                self.B, self.C = coagulation_data['B'], coagulation_data['C']
            else:
                self.B, self.C = compute_B_C(self.coag, self.N, self.Np, self.x_boundaries, self.x_Gauss, self.phi, self.scale_type)
            # Computing coagulation operator:
            self.f_coag = Coagulation_evolution(self).eval
            # Saving coagulation tensors:
            if save_coagulation:
                print('Saving coagulation data...')
                if not coagulation_suffix:
                    data_filename = 'Coagulation_data_Ne=' + str(self.Ne) + '_Np=' + str(self.Np) + '_' + self.scale_type + '.npz'
                else:
                    data_filename = 'Coagulation_data_Ne=' + str(self.Ne) + '_Np=' + str(self.Np) + '_' + self.scale_type + '_' + coagulation_suffix + '.npz'
                pathname = 'C:/Users/Vincent/OneDrive - The University of Auckland/Python/particle_size_distribution_models/evolution_models/data/' + data_filename
                np.savez(pathname, B=self.B, C=self.C)

    # Function to add unknown process models to evolution model:
    def add_unknown(self, process, Ne_process=None, Np_process=None):  # Default for Ne and Np is None
        print('Adding', process, 'as unknown to model...')
        self.unknowns.append(process)
        if process == 'condensation':
            print('Computing condensation as unknown operator...')
            self.Ne_gamma, self.Np_gamma = Ne_process, Np_process
            self.N_gamma = self.Ne_gamma * self.Np_gamma
            if self.scale_type == 'log':
                _, self.x_boundaries_gamma, self.h_gamma = get_discretisation(self.Ne_gamma, self.Np_gamma, self.logDp_lim[0], self.logDp_lim[1])
            else:
                _, self.x_boundaries_gamma, self.h_gamma = get_discretisation(self.Ne_gamma, self.Np_gamma, self.Dp_lim[0], self.Dp_lim[1])
            self.phi_gamma = get_Legendre_basis(self.N_gamma, self.Np_gamma, self.x_boundaries_gamma)
            self.Q_gamma = compute_Q_gamma(self.Ne, self.Np, self.Np_gamma, self.N_gamma, self.phi, self.dphi, self.phi_gamma, self.x_boundaries, self.scale_type)
            self.R1_gamma = compute_R1_gamma(self.Ne, self.Np, self.N_gamma, self.phi, self.phi_gamma, self.x_boundaries, self.scale_type)
            self.R2_gamma = compute_R2_gamma(self.Ne, self.Np, self.N_gamma, self.phi, self.phi_gamma, self.x_boundaries, self.scale_type)
            self.f_cond = Condensation_unknown_evolution(self).eval
        elif process == 'deposition':
            print('Computing deposition as unknown operator...')
            self.Ne_eta, self.Np_eta = Ne_process, Np_process
            self.N_eta = self.Ne_eta * self.Np_eta
            if self.scale_type == 'log':
                _, self.x_boundaries_eta, self.h_eta = get_discretisation(self.Ne_eta, self.Np_eta, self.logDp_lim[0], self.logDp_lim[1])
            else:
                _, self.x_boundaries_eta, self.h_eta = get_discretisation(self.Ne_eta, self.Np_eta, self.Dp_lim[0], self.Dp_lim[1])
            self.phi_eta = get_Legendre_basis(self.N_eta, self.Np_eta, self.x_boundaries_eta)
            self.D_eta = compute_D_eta(self.N, self.Np, self.Np_eta, self.N_eta, self.phi, self.phi_eta, self.x_boundaries, self.scale_type)
            self.f_depo = Deposition_unknown_evolution(self).eval
        elif process == 'source':
            print('Computing source as unknown operator...')
            self.f_sorc = Source_unknown_evolution(self).eval

    # Compilation of model (for example: getting function f such that dalpha(t) / dt = f(x(t), t)):
    def compile(self, **kwargs):
        do_time_integration = get_kwarg_value(kwargs, 'do_time_integration', True)  # Set to True to compile time integrator
        self.f = get_f(self)  # Getting function f such that dalpha(t) / dt = f(x(t), t)
        if do_time_integration:
            self.time_integrator = get_kwarg_value(kwargs, 'time_integrator', None)  # Time integrator for time evolution (e.g. None, 'euler' or 'RK4')
            self.next_step = get_next_step(self)  # Getting function next_step such that alpha_{t + 1} = next_step(x_t, t)

    # Evaluate next time step alpha_{t + 1} = F_t(x_t) or simply dalpha(t) / dt = f(x(t), t)):
    def eval(self, x, t):
        return self.next_step(x, t)

    # Function to compute coefficient(t) from function f(x, t):
    def compute_coefficients(self, coefficient, f):
        if coefficient == 'alpha':
            return compute_coefficients(f, self.N, self.Np, self.phi, self.x_boundaries, self.h)
        elif coefficient == 'gamma':
            return compute_coefficients(f, self.N_gamma, self.Np_gamma, self.phi_gamma, self.x_boundaries_gamma, self.h_gamma)
        elif coefficient == 'eta':
            return compute_coefficients(f, self.N_eta, self.Np_eta, self.phi_eta, self.x_boundaries_eta, self.h_eta)

    # Computing plotting discretisation over [0, T] from all alpha = [alpha_0, alpha_1, ..., alpha_NT]:
    def get_nplot_discretisation(self, alpha, **kwargs):
        print('Computing size distribution plotting discretisation...')
        Gamma_alpha = get_kwarg_value(kwargs, 'Gamma_alpha', np.zeros([self.NT, self.N, self.N]))  # Covariance matrix of alpha
        convert_v_to_Dp = get_kwarg_value(kwargs, 'convert_v_to_Dp', False)  # Set to True to convert from n_v to n_Dp (and covariance)
        convert_x_to_logDp = get_kwarg_value(kwargs, 'convert_x_to_logDp', False)  # Set to True to convert from n_x to n_logDp (and covariance)
        if convert_v_to_Dp:
            d_plot, v_plot, n_v_plot, Gamma_n_v = get_plotting_discretisation(alpha, Gamma_alpha, self.x_boundaries, self.h, self.phi, self.N, self.Ne, self.Np, self.scale_type, return_Gamma=True)
            matrix_v_to_Dp = np.diag((np.pi / 2) * (d_plot ** 2))
            n_Dp_plot, sigma_n_Dp = change_basis_operator(n_v_plot, Gamma_n_v, matrix_v_to_Dp, time_varying=True, return_sigma=True)
            return d_plot, v_plot, n_Dp_plot, sigma_n_Dp
        elif convert_x_to_logDp:
            d_plot, v_plot, n_x_plot, Gamma_n_x = get_plotting_discretisation(alpha, Gamma_alpha, self.x_boundaries, self.h, self.phi, self.N, self.Ne, self.Np, self.scale_type, return_Gamma=True)
            matrix_x_to_logDp = (3 / np.log10(np.e)) * np.eye(len(d_plot))
            n_logDp_plot, sigma_n_logDp = change_basis_operator(n_x_plot, Gamma_n_x, matrix_x_to_logDp, time_varying=True, return_sigma=True)
            return d_plot, v_plot, n_logDp_plot, sigma_n_logDp
        else:
            d_plot, v_plot, n_plot, sigma_n = get_plotting_discretisation(alpha, Gamma_alpha, self.x_boundaries, self.h, self.phi, self.N, self.Ne, self.Np, self.scale_type)
            return d_plot, v_plot, n_plot, sigma_n

    # Computing plotting discretisation parameter over [0, T]:
    def get_parameter_estimation_discretisation(self, process, coefficient, Gamma, **kwargs):
        if process == 'condensation':
            print('Computing condensation plotting discretisation...')
            _, d_plot, cond_plot, sigma_cond = get_plotting_discretisation(coefficient, Gamma, self.x_boundaries_gamma, self.h_gamma, self.phi_gamma, self.N_gamma, self.Ne_gamma, self.Np_gamma, 'linear')
            return _, d_plot, cond_plot, sigma_cond
        elif process == 'deposition':
            print('Computing deposition plotting discretisation...')
            _, d_plot, depo_plot, sigma_depo = get_plotting_discretisation(coefficient, Gamma, self.x_boundaries_eta, self.h_eta, self.phi_eta, self.N_eta, self.Ne_eta, self.Np_eta, 'linear')
            return _, d_plot, depo_plot, sigma_depo
        elif process == 'nucleation':
            print('Computing nucleation plotting discretisation...')
            # Parameters:
            convert_v_to_Dp = get_kwarg_value(kwargs, 'convert_v_to_Dp', False)  # Set to True to convert from v to Dp (and covariance)
            convert_x_to_logDp = get_kwarg_value(kwargs, 'convert_x_to_logDp', False)  # Set to True to convert from x to logDp (and covariance)
            # Equating coefficients to parameter names:
            J = coefficient
            Gamma_J = Gamma
            # Precomputations:
            NT = len(J)
            Dp_min = volume_to_diameter(self.xlim[0])
            if convert_v_to_Dp:
                matrix_v_to_Dp = (np.pi / 2) * (Dp_min ** 2) * np.eye(NT)
                J_Dp, sigma_J_Dp = change_basis_operator(J, Gamma_J, matrix_v_to_Dp, return_sigma=True)
                return J_Dp, sigma_J_Dp
            elif convert_x_to_logDp:
                matrix_x_to_logDp = (3 / np.log10(np.e)) * np.eye(NT)
                J_logDp, sigma_J_logDp = change_basis_operator(J, Gamma_J, matrix_x_to_logDp, return_sigma=True)
                return J_logDp, sigma_J_logDp
            else:
                sigma_J = np.sqrt(Gamma_J)
                return J, sigma_J

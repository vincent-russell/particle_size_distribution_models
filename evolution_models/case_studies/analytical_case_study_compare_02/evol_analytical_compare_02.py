"""

Title: Computes solution approximations to the coagulation equation in log-space.
Author: Vincent Russell
Date: June 7, 2022

"""



#######################################################
# Modules:
import numpy as np
import time as tm
from tkinter import mainloop
from tqdm import tqdm

# Local modules:
import basic_tools
from evolution_models.tools import GDE_evolution_model


#######################################################
if __name__ == '__main__':

    #######################################################
    # Parameters:

    # Setup and plotting:
    N_plot = 12  # Plotting discretisation
    plot_animations = True  # Set to True to plot animations
    load_coagulation = False  # Set to True to load coagulation tensors
    save_coagulation = False  # Set to True to save coagulation tensors
    coagulation_suffix = 'evol_analytical_compare_02'  # Suffix of saved coagulation tensors file

    # Spatial domain:
    vmin = 1e-7  # Minimum volume of particles (micro m^3)
    vmax = 1e-2  # Maximum volume of particles (micro m^3)
    xmin = np.log(vmin)  # Lower limit in log-size
    xmax = np.log(vmax)  # Upper limit in log-size

    # Time domain:
    dt = 0.1  # Time step (hours)
    T = 96  # End time (hours)
    NT = int(T / dt)  # Total number of time steps

    # Size distribution discretisation:
    Ne = 12  # Number of elements
    Np = 2  # Np - 1 = degree of Legendre polynomial approximation in each element
    N = Ne * Np  # Total degrees of freedom

    # Standard FEM size distribution discretisation:
    N_standard = 12  # Number of nodes in finite element mesh

    # Initial condition n_v(v, 0):
    N_0 = 1e4  # Total initial number of particles (particles per cm^3)
    v_0 = 2e-5  # Mean initial volume (micro m^3)
    def initial_condition_v(v):
        return ((N_0 * v) / (v_0 ** 2)) * np.exp(-v / v_0)
    def initial_condition(x):
        v = np.exp(x)
        return v * initial_condition_v(v)

    # Set to True for imposing boundary condition n(vmin, t) = 0:
    boundary_zero = True

    # Coagulation model:
    beta_0 = 8e-6  # Coagulation parameter (cm^3 hour^-1)
    def coag(*_):
        return beta_0


    #######################################################
    # Initialising timer for total computation:
    basic_tools.print_lines()  # Print lines in console
    initial_time = tm.time()  # Initial time stamp


    #######################################################
    # Computing plotting discretisation:
    x_plot = np.linspace(xmin, xmax, N_plot)
    v_plot = np.exp(x_plot)
    d_plot = basic_tools.volume_to_diameter(v_plot)


    # =========================================================#
    # NOTE: The following is the proposed (new) model.
    # =========================================================#


    #######################################################
    # Constructing evolution model:
    print('Constructing proposed model...')
    F = GDE_evolution_model(Ne, Np, xmin, xmax, dt, NT, boundary_zero=boundary_zero, scale_type='log', print_status=False)  # Initialising evolution model
    F.add_process('coagulation', coag, load_coagulation=load_coagulation, save_coagulation=save_coagulation, coagulation_suffix=coagulation_suffix)  # Adding coagulation to evolution model
    F.compile(time_integrator='euler')  # Compiling evolution model and adding time integrator


    #######################################################
    # Computing time evolution of model:
    print('Computing time evolution using proposed model...')
    alpha = np.zeros([N, NT])  # Initialising alpha = [alpha_0, alpha_1, ..., alpha_NT]
    alpha[:, 0] = F.compute_coefficients('alpha', initial_condition)  # Computing alpha coefficients from initial condition function
    t = np.zeros(NT)  # Initialising time array
    for k in tqdm(range(NT - 1)):  # Iterating over time
        alpha[:, k + 1] = F.eval(alpha[:, k], t[k])  # Time evolution computation
        t[k + 1] = (k + 1) * dt  # Time (hours)


    #######################################################
    # Computing plotting discretisation:
    _, _, n_x_plot, _ = F.get_nplot_discretisation(alpha, x_plot=x_plot)  # Computing plotting discretisation


    # =========================================================#
    # NOTE: The following is the standard FEM (or PGFEM) model.
    # =========================================================#
    print('Constructing standard model...')


    #######################################################
    # Function to get piecewise linear basis function:
    def get_basis_function(x_i_minus_1, x_i, x_i_plus_1):
        def basis_function(x):
            if x_i_minus_1 < x < x_i:
                return (x - x_i_minus_1) / (x_i - x_i_minus_1)
            elif x_i < x < x_i_plus_1:
                return (x_i_plus_1 - x) / (x_i_plus_1 - x_i)
            elif x == x_i:
                return 1
            else:
                return 0
        return basis_function


    #######################################################
    # Function to get derivative of piecewise linear basis function:
    def get_derivative_basis_function(x_i_minus_1, x_i, x_i_plus_1):
        def derivative_basis_function(x):
            if x_i_minus_1 < x < x_i:
                return 1 / (x_i - x_i_minus_1)
            elif x_i < x < x_i_plus_1:
                return -1 / (x_i_plus_1 - x_i)
            else:
                return 0
        return derivative_basis_function


    #######################################################
    # Computing basis functions:
    x_standard = np.linspace(xmin, xmax, N_standard)
    v_standard = np.exp(x_standard)
    d_standard = basic_tools.volume_to_diameter(v_standard)
    phi = np.array([])  # Initialising array of basis functions
    dphi = np.array([])  # Initialising array of derivative of basis functions
    for i in range(N_standard):  # Iterating over number of nodes (number of basis functions)
        phi_i = get_basis_function(x_standard[max(i - 1, 0)], x_standard[i], x_standard[min(i + 1, N_standard - 1)])  # Computing basis functions
        dphi_i = get_derivative_basis_function(x_standard[max(i - 1, 0)], x_standard[i], x_standard[min(i + 1, N_standard - 1)])  # Computing derivative of basis functions
        phi = np.append(phi, phi_i)  # Appending i-th basis function to array
        dphi = np.append(dphi, dphi_i)  # Appending i-th basis function to array


    #######################################################
    # Computing M matrix:
    print('Computing M matrix...')
    M = np.zeros([N_standard, N_standard])
    for i in tqdm(range(N_standard)):
        for j in range(N_standard):
            def M_integrand(x):
                return phi[i](x) * phi[j](x)
            M[j, i] = basic_tools.GLnpt(M_integrand, x_standard[max(i - 1, 0)], x_standard[min(i + 1, N_standard - 1)], 8)


    #######################################################
    # Computing or loading B and C tensors:
    if load_coagulation:
        coag_tensors = np.load('coag_tensors_N_standard=' + str(N_standard) + '_' + coagulation_suffix + '.npz')
        B, C = coag_tensors['B'], coag_tensors['C']
    else:
        print('Computing tensors B and C...')
        B = np.zeros([N_standard, N_standard, N_standard])  # Initialising
        C = np.zeros([N_standard, N_standard, N_standard])  # Initialising

        # Iterating over i-th matrix in tensor:
        for i in tqdm(range(N_standard)):
            # Iterating over k-th entries in i-th matrix:
            for k in range(N_standard):
                # Iterating over j-th entries in i-th matrix:
                for j in range(N_standard):

                    def B_integrand(x):

                        # Integrand:
                        def B_sub_integrand(y):
                            xy = np.log(np.exp(x) - np.exp(y))
                            cst = 1 / (np.exp(x) - np.exp(y))
                            return cst * coag(xy, y) * phi[j](xy) * phi[k](y)

                        # Limit check:
                        xlim = np.log(np.exp(x) - np.exp(xmin))
                        if x_standard[min(k + 1, N_standard - 1)] < xlim:
                            return np.exp(x) * phi[i](x) * basic_tools.GLnpt(B_sub_integrand, x_standard[max(k - 1, 0)], x_standard[min(k + 1, N_standard - 1)], 8)
                        elif x_standard[max(k - 1, 0)] <= xlim <= x_standard[min(k + 1, N_standard - 1)]:
                            return np.exp(x) * phi[i](x) * basic_tools.GLnpt(B_sub_integrand, x_standard[max(k - 1, 0)], xlim, 8)
                        else:
                            return 0

                    B[i, j, k] = (1 / 2) * basic_tools.GLnpt(B_integrand, x_standard[max(i - 1, 0)], x_standard[min(i + 1, N_standard - 1)], 8)

                    # Computing C^i_j,k elements:
                    def C_integrand(x):
                        def C_sub_integrand(y):
                            return coag(x, y) * phi[k](y)
                        C_sub_integral = basic_tools.GLnpt(C_sub_integrand, x_standard[max(k - 1, 0)], x_standard[min(k + 1, N_standard - 1)], 8)
                        return phi[j](x) * C_sub_integral * phi[i](x)

                    C[i, j, k] = basic_tools.GLnpt(C_integrand, x_standard[max(i - 1, 0)], x_standard[min(i + 1, N_standard - 1)], 8)

    # Saving coagulation tensors B and C:
    if save_coagulation:
        np.savez('coag_tensors_N_standard=' + str(N_standard) + '_' + coagulation_suffix, B=B, C=C)


    #######################################################
    # Computing evolution model:

    # Coagulation:
    BC = B - C
    def f_coag(n):
        quad_vec = np.zeros(N_standard)
        for i in range(N_standard):
            quad_alpha = np.dot(BC[i], n)
            quad_vec[i] = np.dot(n, quad_alpha)
        return np.linalg.solve(M, quad_vec)

    # Evolution model:
    def F_standard(n):
        return n + dt * f_coag(n)


    #######################################################
    # Computing initial condition:
    n_x_standard = np.zeros([N_standard, NT])  # Initialising n = [n_0, n_1, ..., n_NT]
    for i in range(N_standard):
        n_x_standard[i, 0] = initial_condition(x_standard[i])  # Computing initial condition


    #######################################################
    # Computing time evolution of model:
    print('Computing time evolution using standard model...')
    for k in tqdm(range(NT - 1)):  # Iterating over time
        n_x_standard[:, k + 1] = F_standard(n_x_standard[:, k])  # Time evolution computation


    #######################################################
    # Computing analytical solution:
    print('Computing analytical solution...')
    x_analytical = np.linspace(xmin, xmax, N_plot)  # Log-discretisation for analytical solution
    v_analytical = np.exp(x_analytical)  # Discretisation for analytical solution
    n_v_analytical = np.zeros([len(v_analytical), NT])  # Initialising
    n_x_analytical = np.zeros([len(v_analytical), NT])  # Initialising
    for k in range(1, NT):  # Iterating over time
        M_T = (2 * N_0) / (2 + (beta_0 * N_0 * t[k]))
        N_T = 1 - (M_T / N_0)
        for i in range(len(v_analytical)):  # Iterating over volume
            A = ((1 - N_T) ** 2 / np.sqrt(N_T)) * (N_0 / v_0)
            B = v_analytical[i] / v_0
            C = (v_analytical[i] * np.sqrt(N_T)) / v_0
            n_v_analytical[i, k] = A * np.exp(-B) * np.sinh(C)
            n_x_analytical[i, k] = n_v_analytical[i, k] * v_analytical[i]
    n_v_analytical = np.nan_to_num(n_v_analytical)  # Replace NaN with zeros
    n_x_analytical = np.nan_to_num(n_x_analytical)  # Replace NaN with zeros


    #######################################################
    # Computing total error (l2 norm):
    n_diff_proposed = n_x_analytical - n_x_plot
    n_diff_standard = n_x_analytical - n_x_standard
    norm_diff_proposed = np.zeros(NT)  # Initialising
    norm_diff_standard = np.zeros(NT)  # Initialising
    for k in range(NT):
        norm_diff_proposed[k] = np.sqrt(np.matmul(n_diff_proposed[:, k], n_diff_proposed[:, k]))
        norm_diff_standard[k] = np.sqrt(np.matmul(n_diff_standard[:, k], n_diff_standard[:, k]))
    total_error_proposed = np.sum(norm_diff_proposed)
    total_error_standard = np.sum(norm_diff_standard)
    print('Total error of proposed model:', round(total_error_proposed))
    print('Total error of standard model:', round(total_error_standard))


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
    xticks = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]  # Plot x-tick labels
    xticklabels = ['$10^{-7}$', '$10^{-6}$', '$10^{-5}$', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$']  # Plot x-tick labels
    xlimits = [vmin, vmax]  # Plot boundary limits for x-axis
    ylimits = [0, 6000]  # Plot boundary limits for y-axis
    xlabel = '$v$ ($\mu$m$^3$)'  # x-axis label for 1D animation plot
    ylabel = '$\dfrac{dN}{d\ln(v)}$ (cm$^{-3})$'  # y-axis label for 1D animation plot
    title = 'Size distribution'  # Title for 1D animation plot
    legend = ['Analytical solution', 'Collocation', 'FEM']  # Adding legend to plot
    line_color = ['green', 'blue', 'red']  # Colors of lines in plot
    line_style = ['solid', 'dashed', 'dotted']  # Style of lines in plot
    time = t  # Array where time[i] is plotted (and animated)
    timetext = ('Time = ', ' hours')  # Tuple where text to be animated is: timetext[0] + 'time[i]' + timetext[1]
    delay = 0  # Delay between frames in milliseconds

    # Size distribution animation:
    basic_tools.plot_1D_animation(v_plot, n_x_analytical, n_x_plot, plot_add=(v_standard, n_x_standard), xticks=xticks, xticklabels=xticklabels, xlimits=xlimits, ylimits=ylimits, xscale=xscale, xlabel=xlabel, ylabel=ylabel, title=title,
                                  delay=delay, location=location, legend=legend, time=time, timetext=timetext, line_color=line_color, line_style=line_style, doing_mainloop=False)

    # Mainloop and print:
    if plot_animations:
        print('Plotting animations...')
        mainloop()  # Runs tkinter GUI for plots and animations

    # Final print statements
    basic_tools.print_lines()  # Print lines in console
    print()  # Print space in console


"""

Title: Computes solution approximations to the condensation equation in log-space
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
from evolution_models.tools import GDE_evolution_model, change_basis_x_to_logDp, change_basis_ln_to_linear, change_basis_volume_to_diameter


#######################################################
if __name__ == '__main__':

    #######################################################
    # Parameters:

    # Setup and plotting:
    N_plot = 50  # Plotting discretisation
    plot_animations = True  # Set to True to plot animations

    # Spatial domain:
    Dp_min = 0.1  # Minimum diameter of particles (micro m)
    Dp_max = 1  # Maximum diameter of particles (micro m)
    vmin = basic_tools.diameter_to_volume(Dp_min)  # Minimum volume of particles (micro m^3)
    vmax = basic_tools.diameter_to_volume(Dp_max)  # Maximum volume of particles (micro m^3)
    xmin = np.log(vmin)  # Lower limit in log-size
    xmax = np.log(vmax)  # Upper limit in log-size

    # Time domain:
    dt = 0.05  # Time step (hours)
    T = 96  # End time (hours)
    NT = int(T / dt)  # Total number of time steps

    # Size distribution discretisation:
    Ne = 50  # Number of elements
    Np = 2  # Np - 1 = degree of Legendre polynomial approximation in each element
    N = Ne * Np  # Total degrees of freedom

    # Standard FEM size distribution discretisation:
    N_standard = 50  # Number of nodes in finite element mesh

    # PGFEM parameters:
    do_pgfem = True  # Set to True to do PGFEM, else False for FEM
    epsilon = 0.2  # Upwinding factor

    # Initial condition n_Dp(Dp, 0):
    N_0 = 180  # Amplitude of initial condition gaussian
    d_mean = 0.3  # Mean of initial condition gaussian
    sigma_g = 1.2  # Standard deviation of initial condition gaussian
    def initial_condition_Dp(Dp):
        amp = N_0 / (np.sqrt(2 * np.pi) * Dp * np.log(sigma_g))
        cst = (np.log(Dp / d_mean) ** 2) / (2 * np.log(sigma_g) ** 2)
        return amp * np.exp(-cst)
    def initial_condition(x):
        v = np.exp(x)
        Dp = basic_tools.volume_to_diameter(v)
        cst = v * (2 / (np.pi * Dp ** 2))
        return cst * initial_condition_Dp(Dp)

    # Set to True for imposing boundary condition n(vmin, t) = 0:
    boundary_zero = True

    # Condensation model I_Dp(Dp, t):
    A = 0.001   # Condensation parameter (micro m^2 hour^{-1})
    def cond(Dp):
        return A / Dp


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
    F.add_process('condensation', cond)  # Adding condensation to evolution model
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
    n_v_plot = change_basis_ln_to_linear(n_x_plot, v_plot)
    n_Dp_plot = change_basis_volume_to_diameter(n_v_plot, d_plot)
    n_logDp_plot = change_basis_x_to_logDp(n_x_plot, v_plot, d_plot)


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
    # Function to get sigma modifier for PGFEM basis functions:
    def get_sigma_modifier(x_i_minus_1, x_i):
        h_i = x_i - x_i_minus_1
        def sigma_modifier(x):
            if x_i_minus_1 < x < x_i:
                return (4 / (h_i ** 2)) * (x - x_i_minus_1) * (x_i - x)
            else:
                return 0
        return sigma_modifier


    #######################################################
    # Function to get derivative of sigma modifier for PGFEM basis functions:
    def get_derivative_sigma_modifier(x_i_minus_1, x_i):
        h_i = x_i - x_i_minus_1
        def derivative_sigma_modifier(x):
            if x_i_minus_1 < x < x_i:
                return (4 / (h_i ** 2)) * (x_i + x_i_minus_1 - 2 * x)
            else:
                return 0
        return derivative_sigma_modifier


    #######################################################
    # Function to get PGFEM basis functions:
    def get_pgfem_basis_functions(x_i_minus_1, x_i, x_i_plus_1, epsilon):
        fem_basis_function_i = get_basis_function(x_i_minus_1, x_i, x_i_plus_1)
        sigma_modifier_i = get_sigma_modifier(x_i_minus_1, x_i)
        sigma_modifier_i_plus_1 = get_sigma_modifier(x_i, x_i_plus_1)
        def pgfem_basis_function(x):
            return fem_basis_function_i(x) + (3 / 2) * epsilon * (sigma_modifier_i(x) - sigma_modifier_i_plus_1(x))
        return pgfem_basis_function


    #######################################################
    # Function to get derivative of PGFEM basis functions:
    def get_derivative_pgfem_basis_functions(x_i_minus_1, x_i, x_i_plus_1, epsilon):
        derivative_fem_basis_function_i = get_derivative_basis_function(x_i_minus_1, x_i, x_i_plus_1)
        derivative_sigma_modifier_i = get_derivative_sigma_modifier(x_i_minus_1, x_i)
        derivative_sigma_modifier_i_plus_1 = get_derivative_sigma_modifier(x_i, x_i_plus_1)
        def derivative_pgfem_basis_function(x):
            return derivative_fem_basis_function_i(x) + (3 / 2) * epsilon * (derivative_sigma_modifier_i(x) - derivative_sigma_modifier_i_plus_1(x))
        return derivative_pgfem_basis_function


    #######################################################
    # Computing basis functions:
    x_standard = np.linspace(xmin, xmax, N_standard)
    v_standard = np.exp(x_standard)
    d_standard = basic_tools.volume_to_diameter(v_standard)
    phi = np.array([])  # Initialising array of basis functions
    dphi = np.array([])  # Initialising array of derivative of basis functions
    for i in range(N_standard):  # Iterating over number of nodes (number of basis functions)
        if do_pgfem:
            phi_i = get_pgfem_basis_functions(x_standard[max(i - 1, 0)], x_standard[i], x_standard[min(i + 1, N_standard - 1)], epsilon)  # Computing basis functions
            dphi_i = get_derivative_pgfem_basis_functions(x_standard[max(i - 1, 0)], x_standard[i], x_standard[min(i + 1, N_standard - 1)], epsilon)  # Computing derivative of basis functions
        else:
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
            M[j, i] = basic_tools.GLnpt(M_integrand, x_standard[max(i - 1, 0)], x_standard[min(i + 1, N_standard - 1)], 30)


    #######################################################
    # Computing Q matrix:
    print('Computing Q matrix...')
    Q = np.zeros([N_standard, N_standard])
    for i in tqdm(range(N_standard)):
        for j in range(N_standard):
            def Q_integrand(x):
                v = np.exp(x)
                Dp = basic_tools.volume_to_diameter(v)
                return (3 / Dp) * cond(Dp) * phi[i](x) * dphi[j](x)
            Q[j, i] = basic_tools.GLnpt(Q_integrand, x_standard[max(i - 1, 0)], x_standard[min(i + 1, N_standard - 1)], 30)


    #######################################################
    # Implementing boundary condition n(xmin, t) = 0:
    M[0, :] = 0; M[:, 0] = 0; M[0, 0] = 1
    Q[0, :] = 0; Q[:, 0] = 0


    #######################################################
    # Computing evolution model:

    # Condensation:
    R = np.linalg.solve(M, Q)
    def f_cond(n):
        return np.matmul(R, n)

    # Evolution model:
    def F_standard(n):
        return n + dt * f_cond(n)


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
    # Computing plotting discretisation:
    n_v_standard = change_basis_ln_to_linear(n_x_standard, v_standard)
    n_Dp_standard = change_basis_volume_to_diameter(n_v_standard, d_standard)
    n_logDp_standard = change_basis_x_to_logDp(n_x_standard, v_standard, d_standard)


    #######################################################
    # Computing analytical solution:
    print('Computing analytical solution...')
    x_analytical = np.linspace(xmin, xmax, N_plot)  # Log-discretisation
    v_analytical = np.exp(x_analytical)  # Volume discretisation
    Dp_analytical = basic_tools.volume_to_diameter(v_analytical)  # Diameter discretisation
    n_Dp_analytical = np.zeros([len(v_analytical), NT])  # Initialising
    n_logDp_analytical = np.zeros([len(v_analytical), NT])  # Initialising
    for k in range(1, NT):  # Iterating over time
        for i in range(len(v_analytical)):  # Iterating over volume
            cst = (Dp_analytical[i] ** 2) - (2 * A * t[k])
            B = (Dp_analytical[i] * N_0) / (np.sqrt(2 * np.pi) * np.log(sigma_g) * cst)
            C = (np.log(np.sqrt(cst) / d_mean) ** 2) / (2 * np.log(sigma_g) ** 2)
            n_Dp_analytical[i, k] = B * np.exp(-C)
            n_logDp_analytical[i, k] = (1 / np.log10(np.e)) * Dp_analytical[i] * n_Dp_analytical[i, k]
    n_Dp_analytical = np.nan_to_num(n_Dp_analytical)  # Replace NaN with zeros
    n_logDp_analytical = np.nan_to_num(n_logDp_analytical)  # Replace NaN with zeros


    #######################################################
    # Computing total error (l2 norm):
    n_diff_proposed = n_logDp_analytical - n_logDp_plot
    n_diff_standard = n_logDp_analytical - n_logDp_standard
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
    xticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  # Plot x-tick labels
    xlimits = [d_plot[0], d_plot[-1]]  # Plot boundary limits for x-axis
    ylimits = [-500, 4000]  # Plot boundary limits for y-axis
    xlabel = '$D_p$ ($\mu$m)'  # x-axis label for 1D animation plot
    ylabel = '$\dfrac{dN}{dlogD_p}$ (cm$^{-3})$'  # y-axis label for 1D animation plot
    title = 'Size distribution'  # Title for 1D animation plot
    if do_pgfem:
        legend = ['Analytical solution', 'DGFEM', 'PGFEM']  # Adding legend to plot
    else:
        legend = ['Analytical solution', 'DGFEM', 'FEM']  # Adding legend to plot
    legend_position = 'upper left'  # Position of legend
    line_color = ['green', 'blue', 'red']  # Colors of lines in plot
    line_style = ['solid', 'dashed', 'dotted']  # Style of lines in plot
    time = t  # Array where time[i] is plotted (and animated)
    timetext = ('Time = ', ' hours')  # Tuple where text to be animated is: timetext[0] + 'time[i]' + timetext[1]
    delay = 0  # Delay between frames in milliseconds

    # Size distribution animation:
    basic_tools.plot_1D_animation(d_plot, n_logDp_analytical, n_logDp_plot, plot_add=(d_standard, n_logDp_standard), xticks=xticks, xlimits=xlimits, ylimits=ylimits, xscale=xscale, xlabel=xlabel, ylabel=ylabel, title=title,
                                  delay=delay, location=location, time=time, timetext=timetext, line_color=line_color, line_style=line_style, legend=legend, legend_position=legend_position, doing_mainloop=False)

    # Mainloop and print:
    if plot_animations:
        print('Plotting animations...')
        mainloop()  # Runs tkinter GUI for plots and animations

    # Final print statements
    basic_tools.print_lines()  # Print lines in console
    print()  # Print space in console


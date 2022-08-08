"""

Title: Computes solution approximations to the general dynamic equation in log-space.
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
from evolution_models.tools import Fuchs_Brownian, change_basis_volume_to_diameter, change_basis_volume_to_diameter_sorc


#######################################################
if __name__ == '__main__':

    #######################################################
    # Parameters:

    # Setup and plotting:
    plot_animations = True  # Set to True to plot animations
    plot_images = False  # Set to True to plot images
    load_coagulation = True  # Set to True to load coagulation tensors
    save_coagulation = False  # Set to True to save coagulation tensors

    # Spatial domain:
    Dp_min = 1  # Minimum diameter of particles (micro m)
    Dp_max = 10  # Maximum diameter of particles (micro m)
    vmin = basic_tools.diameter_to_volume(Dp_min)  # Minimum volume of particles (micro m^3)
    vmax = basic_tools.diameter_to_volume(Dp_max)  # Maximum volume of particles (micro m^3)

    # Time domain:
    dt = (1 / 60) * 2  # Time step (hours)
    T = 12  # End time (hours)
    NT = int(T / dt)  # Total number of time steps

    # Size distribution discretisation:
    N = 50  # Number of nodes in finite element mesh

    # Initial condition n_0(v) = n(v, 0):
    N_0 = 300  # Amplitude of initial condition gaussian
    v_0 = basic_tools.diameter_to_volume(4)  # Mean of initial condition gaussian
    sigma_0 = 15  # Standard deviation of initial condition gaussian
    def initial_condition(v):
        return basic_tools.gaussian(v, N_0, v_0, sigma_0)

    # Set to True for imposing boundary condition n(vmin, t) = 0:
    boundary_zero = True

    # Condensation model I_Dp(Dp, t):
    I_0 = 0*0.2  # Condensation parameter constant
    I_1 = 0*1  # Condensation parameter inverse quadratic
    def cond(Dp):
        return I_0 + I_1 / (Dp ** 2)

    # Deposition model d(Dp, t):
    depo_Dpmin = 5  # Deposition parameter; diameter at which minimum
    d_0 = 0.4  # Deposition parameter constant
    d_1 = -0.15  # Deposition parameter linear
    d_2 = -d_1 / (2 * depo_Dpmin)  # Deposition parameter quadratic
    def depo(Dp):
        return d_0 + d_1 * Dp + d_2 * Dp ** 2

    # Source (nucleation event) model:
    N_s = 5e3  # Amplitude of gaussian nucleation event
    t_s = 8  # Mean time of gaussian nucleation event
    sigma_s = 2  # Standard deviation time of gaussian nucleation event
    def sorc(t):  # Source (nucleation) at vmin
        return basic_tools.gaussian(t, N_s, t_s, sigma_s)  # Gaussian source (nucleation event) model output

    # Coagulation model:
    def coag(v_x, v_y):
        Dp_x = basic_tools.volume_to_diameter(v_x)  # Diameter of particle x (micro m)
        Dp_y = basic_tools.volume_to_diameter(v_y)  # Diameter of particle y (micro m)
        return Fuchs_Brownian(Dp_x, Dp_y)


    #######################################################
    # Initialising timer for total computation:
    basic_tools.print_lines()  # Print lines in console
    initial_time = tm.time()  # Initial time stamp


    #######################################################
    # Function to get piecewise linear basis function:
    def get_basis_function(v_i_minus_1, v_i, v_i_plus_1):
        def basis_function(v):
            if v_i_minus_1 < v < v_i:
                return (v - v_i_minus_1) / (v_i - v_i_minus_1)
            elif v_i < v < v_i_plus_1:
                return (v_i_plus_1 - v) / (v_i_plus_1 - v_i)
            elif v == v_i:
                return 1
            else:
                return 0
        return basis_function


    #######################################################
    # Function to get derivative of piecewise linear basis function:
    def get_derivative_basis_function(v_i_minus_1, v_i, v_i_plus_1):
        def derivative_basis_function(v):
            if v_i_minus_1 < v < v_i:
                return 1 / (v_i - v_i_minus_1)
            elif v_i < v < v_i_plus_1:
                return -1 / (v_i_plus_1 - v_i)
            else:
                return 0
        return derivative_basis_function


    #######################################################
    # Computing basis functions:
    v_disc = np.linspace(vmin, vmax, N)  # Discretisation of domain
    phi = np.array([])  # Initialising array of basis functions
    dphi = np.array([])  # Initialising array of derivative of basis functions
    for i in range(N):  # Iterating over number of nodes (number of basis functions)
        phi_i = get_basis_function(v_disc[max(i - 1, 0)], v_disc[i], v_disc[min(i + 1, N - 1)])  # Computing basis functions
        dphi_i = get_derivative_basis_function(v_disc[max(i - 1, 0)], v_disc[i], v_disc[min(i + 1, N - 1)])  # Computing derivative of basis functions
        phi = np.append(phi, phi_i)  # Appending i-th basis function to array
        dphi = np.append(dphi, dphi_i)  # Appending i-th basis function to array


    #######################################################
    # Computing M matrix:
    print()
    print('Computing M matrix...')
    M = np.zeros([N, N])
    for i in tqdm(range(N)):
        for j in range(N):
            def M_integrand(v):
                return phi[i](v) * phi[j](v)
            M[j, i] = basic_tools.GLnpt(M_integrand, v_disc[max(i - 1, 0)], v_disc[min(i + 1, N - 1)], 8)


    #######################################################
    # Computing Q matrix:
    print()
    print('Computing Q matrix...')
    Q = np.zeros([N, N])
    for i in tqdm(range(N)):
        for j in range(N):
            def Q_integrand(v):
                Dp = basic_tools.volume_to_diameter(v)
                return (np.pi / 2) * (Dp ** 2) * cond(Dp) * phi[i](v) * dphi[j](v)
            Q[j, i] = basic_tools.GLnpt(Q_integrand, v_disc[max(i - 1, 0)], v_disc[min(i + 1, N - 1)], 8)


    #######################################################
    # Computing or loading B and C tensors:
    if load_coagulation:
        coag_tensors = np.load('coag_tensors_N=' + str(N) + '.npz')
        B, C = coag_tensors['B'], coag_tensors['C']
    else:
        print()
        print('Computing tensors B and C...')
        B = np.zeros([N, N, N])  # Initialising
        C = np.zeros([N, N, N])  # Initialising

        # Iterating over i-th matrix in tensor:
        for i in tqdm(range(N)):
            # Iterating over k-th entries in i-th matrix:
            for k in range(N):
                # Iterating over j-th entries in i-th matrix:
                for j in range(N):

                    def B_integrand(v):

                        # Integrand:
                        def B_sub_integrand(w):
                            return coag(v - w, w) * phi[j](v - w) * phi[k](w)

                        # Limit check:
                        vlim = v - vmin
                        if v_disc[min(k + 1, N - 1)] < vlim:
                            return phi[i](v) * basic_tools.GLnpt(B_sub_integrand, v_disc[max(k - 1, 0)], v_disc[min(k + 1, N - 1)], 8)
                        elif v_disc[max(k - 1, 0)] <= vlim <= v_disc[min(k + 1, N - 1)]:
                            return phi[i](v) * basic_tools.GLnpt(B_sub_integrand, v_disc[max(k - 1, 0)], vlim, 8)
                        else:
                            return 0

                    B[i, j, k] = (1 / 2) * basic_tools.GLnpt(B_integrand, v_disc[max(i - 1, 0)], v_disc[min(i + 1, N - 1)], 8)

                    # Computing C^i_j,k elements:
                    def C_integrand(v):
                        def C_sub_integrand(w):
                            return coag(v, w) * phi[k](w)
                        C_sub_integral = basic_tools.GLnpt(C_sub_integrand, v_disc[max(k - 1, 0)], v_disc[min(k + 1, N - 1)], 8)
                        return phi[j](v) * C_sub_integral * phi[i](v)

                    C[i, j, k] = basic_tools.GLnpt(C_integrand, v_disc[max(i - 1, 0)], v_disc[min(i + 1, N - 1)], 8)

    # Saving coagulation tensors B and C:
    if save_coagulation:
        np.savez('coag_tensors_N=' + str(N), B=B, C=C)


    #######################################################
    # Computing evolution model:

    # Condensation:
    R = np.linalg.solve(M, Q)
    def f_cond(n):
        return np.matmul(R, n)

    # Coagulation:
    BC = B - C
    def f_coag(n):
        quad_vec = np.zeros(N)
        for i in range(N):
            quad_alpha = np.dot(BC[i], n)
            quad_vec[i] = np.dot(n, quad_alpha)
        return np.linalg.solve(M, quad_vec)

    # Evolution model:
    def F(n):
        return n + dt * (f_cond(n) + f_coag(n))


    #######################################################
    # Computing initial condition:
    n_v = np.zeros([N, NT])  # Initialising n = [n_0, n_1, ..., n_NT]
    for i in range(N):
        n_v[i, 0] = initial_condition(v_disc[i])  # Computing initial condition


    #######################################################
    # Computing time evolution of model:
    print()
    print('Computing time evolution...')
    t = np.zeros(NT)  # Initialising time array
    for k in tqdm(range(NT - 1)):  # Iterating over time
        n_v[:, k + 1] = F(n_v[:, k])  # Time evolution computation
        t[k + 1] = (k + 1) * dt  # Time (hours)


    #######################################################
    # Computing parameters plotting discretisation:
    d_plot = basic_tools.volume_to_diameter(v_disc)  # Diameter discretisation
    Nplot = len(d_plot)  # Length of size discretisation
    cond_Dp_plot = np.zeros([Nplot, NT])  # Initialising volume-based condensation rate
    depo_plot = np.zeros([Nplot, NT])  # Initialising deposition rate
    sorc_v_plot = np.zeros(NT)  # Initialising volume-based source (nucleation) rate
    for k in range(NT):
        sorc_v_plot[k] = sorc(t[k])  # Computing volume-based nucleation rate
        for i in range(Nplot):
            cond_Dp_plot[i, k] = cond(d_plot[i])  # Computing volume-based condensation rate
            depo_plot[i, k] = depo(d_plot[i])  # Computing deposition rate
    sorc_Dp_plot = change_basis_volume_to_diameter_sorc(sorc_v_plot, Dp_min)  # Computing diameter-based nucleation rate


    #######################################################
    # Computing plotting discretisation:
    n_Dp = change_basis_volume_to_diameter(n_v, d_plot)  # Computing diameter-based size distribution


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
    ylimits = [0, 10000]  # Plot boundary limits for y-axis
    xlabel = '$D_p$ ($\mu$m)'  # x-axis label for 1D animation plot
    ylabel = '$\dfrac{dN}{dD_p}$ $(\mu$m$^{-1}$cm$^{-3})$'  # y-axis label for 1D animation plot
    title = 'Size distribution'  # Title for 1D animation plot
    line_color = ['blue']  # Colors of lines in plot
    time = t  # Array where time[i] is plotted (and animated)
    timetext = ('Time = ', ' hours')  # Tuple where text to be animated is: timetext[0] + 'time[i]' + timetext[1]
    delay = 0  # Delay between frames in milliseconds

    # Parameters for condensation plot:
    ylimits_cond = [0, 2]  # Plot boundary limits for y-axis
    xlabel_cond = '$D_p$ ($\mu$m)'  # x-axis label for plot
    ylabel_cond = '$I(D_p)$ ($\mu$m hour$^{-1}$)'  # y-axis label for plot
    title_cond = 'Condensation rate'  # Title for plot
    location_cond = location + '2'  # Location for plot
    line_color_cond = ['blue']  # Colors of lines in plot

    # Parameters for deposition plot:
    ylimits_depo = [0, 0.6]  # Plot boundary limits for y-axis
    xlabel_depo = '$D_p$ ($\mu$m)'  # x-axis label for plot
    ylabel_depo = '$d(D_p)$ (hour$^{-1}$)'  # y-axis label for plot
    title_depo = 'Deposition rate'  # Title for plot
    location_depo = location + '3'  # Location for plot
    line_color_depo = ['blue']  # Colors of lines in plot

    # Size distribution animation:
    basic_tools.plot_1D_animation(d_plot, n_Dp, xlimits=xlimits, ylimits=ylimits, xscale=xscale, xlabel=xlabel, ylabel=ylabel, title=title,
                                  delay=delay, location=location, time=time, timetext=timetext, line_color=line_color, doing_mainloop=False)

    # Condensation rate animation:
    basic_tools.plot_1D_animation(d_plot, cond_Dp_plot, xlimits=xlimits, ylimits=ylimits_cond, xscale=xscale, xlabel=xlabel_cond, ylabel=ylabel_cond, title=title_cond,
                                  location=location_cond, time=time, timetext=timetext, line_color=line_color_cond, doing_mainloop=False)

    # Mainloop and print:
    if plot_animations:
        print('Plotting animations...')
        mainloop()  # Runs tkinter GUI for plots and animations


    #######################################################
    # Images:

    # Parameters for size distribution images:
    yscale_image = 'linear'  # Change scale of y-axis (linear or log)
    xlabel_image = 'Time (hours)'  # x-axis label for image
    ylabel_image = '$D_p$ ($\mu$m)'  # y-axis label for image
    ylabelcoords = (-0.06, 1.05)  # y-axis label coordinates
    title_image = 'Size distribution'  # Title for image
    image_min = 100  # Minimum of image colour
    image_max = 10000  # Maximum of image colour
    cmap = 'jet'  # Colour map of image
    cbarlabel = '$\dfrac{dN}{dD_p}$ $(\mu$m$^{-1}$cm$^{-3})$'  # Label of colour bar
    cbarticks = [100, 1000, 10000]  # Ticks of colorbar

    # Plotting images:
    if plot_images:
        print('Plotting images...')
        basic_tools.image_plot(time, d_plot, n_Dp, xlabel=xlabel_image, ylabel=ylabel_image, title=title_image,
                               yscale=yscale_image, ylabelcoords=ylabelcoords, image_min=image_min, image_max=image_max, cmap=cmap, cbarlabel=cbarlabel, cbarticks=cbarticks)


    # Final print statements
    basic_tools.print_lines()  # Print lines in console
    print()  # Print space in console

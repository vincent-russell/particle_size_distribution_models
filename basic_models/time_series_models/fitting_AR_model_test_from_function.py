"""

Title: Estimating AR(p) coefficients from simulated data
Author: Vincent Russell
Date: August 10, 2021

"""


#######################################################
# Modules:
import numpy as np
import time as tm
import matplotlib.pyplot as plt
from tqdm import tqdm

# Local modules:
import basic_tools


#######################################################
if __name__ == '__main__':

    #######################################################
    # Parameters:

    # Autoregressive model parameters:
    p = 5  # Order of model, i.e. AR(p)

    # Time parameters:
    dt = 0.05  # Time step (hours)
    T = 24  # End time (hours)
    NT = int(T / dt)  # Total number of time steps

    # Initial condition:
    x_0 = 0

    # Function to approximate by an AR(p) model:
    Amp = 1  # Amplitude of gaussian
    mean = 5  # Mean time of gaussian
    sigma = 0.5  # Standard deviation time of gaussian
    def function(t):
        return basic_tools.gaussian(t, Amp, mean, sigma)  # Gaussian source model output

    # Standard deviation of simulated data from function:
    sigma_e = 0.0001 * Amp


    #######################################################
    # Initialising timer for total computation:
    print(), basic_tools.print_lines(), print('Initialising...')
    initial_time = tm.time()  # Initial time stamp


    #######################################################
    # Constructing state and adding initial state:
    x_simulated = np.zeros(NT)  # Initialising
    x_simulated[0] = x_0  # Adding initial state


    #######################################################
    # Simulating time series x_t for t = 0, 1, ..., NT - 1:
    print('Simulating time series...')
    t = np.zeros(NT)  # Initialising time array
    for k in tqdm(range(NT - 1)):
        x_simulated[k + 1] = function(t[k]) + np.random.normal(scale=sigma_e)
        t[k + 1] = (k + 1) * dt  # Time (hours)


    #######################################################
    # Constructing matrix Phi such that y = Phi * x, where x = [a1, a2]:
    print('Constructing matrix Phi...')
    Phi = np.zeros([NT - p, p])  # Initialising
    for k in range(NT - p):
        for i in range(p):
            Phi[k, i] = x_simulated[k + 1 - i]


    #######################################################
    # Constructing vector y such that y = Phi * x:
    print('Constructing vector y...')
    y = np.zeros(NT - p)
    for k in range(NT - p):
        y[k] = x_simulated[k + 2]


    #######################################################
    # Estimating A_coef = min_x ||y = Phi * x||, where where x = [a1, a2]:
    print('Estimating A1 and A2 from simulated time series...')
    Phi_T = np.transpose(Phi)
    inv_Phi_T_Phi = np.linalg.inv(np.matmul(Phi_T, Phi))
    Phi_T_y = np.matmul(Phi_T, y)
    A_estimate = np.matmul(inv_Phi_T_Phi, Phi_T_y)


    #######################################################
    # Printing total computation time:
    computation_time = round(tm.time() - initial_time, 3)  # Initial time stamp
    print('Total computation time:', str(computation_time), 'seconds.')  # Print statement
    basic_tools.print_lines()


    #######################################################
    # Printing results:
    for i in range(p):
        print(f'a{i + 1} estimate:', round(A_estimate[i], 3))
    basic_tools.print_lines()


    #######################################################
    # Plotting simulated time series:
    print('Plotting...')

    # Simulated state x_t:
    plt.figure(1, figsize=(6.6, 5.6))
    plt.plot(t, x_simulated[:])
    plt.title('Simulated state $x_t$', fontsize=14)
    plt.xlim([0, T])
    plt.ylim([0 - 0.2 * Amp, Amp + 0.2 * Amp])
    plt.ylabel('$x_t$', fontsize=14)
    plt.xlabel('$t$', fontsize=14)
    plt.grid()
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(5, 635, 660, 560)

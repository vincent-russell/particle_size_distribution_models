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
    p = 6  # Order of model, i.e. AR(p)
    NT = 300000  # Total number of time steps
    x_0 = 0.1  # Initial state
    sigma_e = 1  # Standard deviation

    # Autoregressive coefficients x_t+1 = a1 x_t + a2 x_t-1 + ... + e_t:
    a1 = 0.25
    a2 = 0.2
    a3 = 0.175
    a4 = 0.15
    a5 = 0.125
    a6 = 0.099
    a = np.array([a1, a2, a3, a4, a5, a6])  # Vector of AR(p) coefficients


    #######################################################
    # Initialising timer for total computation:
    print(), basic_tools.print_lines(), print('Initialising...')
    initial_time = tm.time()  # Initial time stamp


    #######################################################
    # Creating matrix A:
    print('Constructing matrix A...')
    A = np.zeros([p, p])
    for i in range(p):
        A[0, i] = a[i]
    A = A + np.eye(p, p, -1)  # Adding ones to off-diagonal


    #######################################################
    # Checking eigenvalues are less than 1:
    print('Computing eigenvalues of matrix A...')
    eig_A = np.linalg.eig(A)[0]
    np.set_printoptions(precision=3)
    print('Eigenvalues of A:', eig_A)
    if (eig_A > 1).any():
        print('|An eigenvalue of A| > 1. AR(2) model is not stable.')
        basic_tools.print_lines()
        exit()


    #######################################################
    # Constructing state and adding initial state:
    x_simulated = np.zeros(NT)  # Initialising
    for i in range(p):
        x_simulated[i] = x_0  # Adding initial state


    #######################################################
    # Simulating time series x_t for t = 0, 1, ..., NT - 1:
    print('Simulating time series...')
    for k in tqdm(range(p - 1, NT - 1)):
        AR_sum = 0  # Initialising sum x_t+1 = a1 x_t + a2 x_t-1 + ... + e_t
        for i in range(p):
            AR_sum += a[i] * x_simulated[k - i]
        # Adding noise:
        x_simulated[k + 1] = AR_sum + np.random.normal(scale=sigma_e)


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
        print(f'a{i + 1}:         ', a[i]), print(f'a{i + 1} estimate:', round(A_estimate[i], 3))
    basic_tools.print_lines()


    #######################################################
    # Plotting simulated time series:
    print('Plotting...')

    # Simulated state x_t:
    plt.figure(1, figsize=(6.6, 5.6))
    plt.plot(x_simulated[:])
    plt.title('Simulated state $x_t$', fontsize=14)
    plt.xlim([0, NT])
    plt.ylim([np.min(x_simulated), np.max(x_simulated)])
    plt.ylabel('$x_t$', fontsize=14)
    plt.xlabel('$t$', fontsize=14)
    plt.grid()
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(5, 635, 660, 560)

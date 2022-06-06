"""

Title: Estimating VAR(2) coefficients from simulated data (in R^2)
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
    N = 2  # Dimension of state (x_t in R^N)
    Np = 2  # Order of model, i.e. AR(Np)
    NT = 300  # Total number of time steps
    x_0 = np.array([0, 0])  # Initial state (in R^2)
    sigma_e = np.array([1, 100])  # Standard deviation (in R^2)

    # Matrix A1 elements:
    a1_11 = 0.3
    a1_12 = 0.2
    a1_21 = 0.3
    a1_22 = 0.2

    # Matrix A2 elements:
    a2_11 = 0.2
    a2_12 = 0.3
    a2_21 = 0.2
    a2_22 = 0.3


    #######################################################
    # Initialising timer for total computation:
    print(), basic_tools.print_lines(), print('Initialising...')
    initial_time = tm.time()  # Initial time stamp


    #######################################################
    # Creating matrix A:
    print('Constructing matrices A1, A2, and matrix A...')
    A1 = np.array([[a1_11, a1_12], [a1_21, a1_22]])
    A2 = np.array([[a2_11, a2_12], [a2_21, a2_22]])
    A = np.zeros([4, 4])
    A[0: 2, 0: 2] = A1
    A[0: 2, 2: 4] = A2
    A[2: 4, 0: 2] = np.eye(2)


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
    # Simulating time series x_t for t = 0, 1, ..., NT - 1:
    print('Simulating time series...')
    x_simulated = np.zeros([2, NT])  # Initialising
    e_simulated = np.zeros(2)  # Initialising
    x_simulated[0: 2, 0] = x_0  # Adding initial state
    for t in tqdm(range(1, NT - 1)):
        e_simulated[0: 2] = np.random.normal(scale=sigma_e)
        x_simulated[:, t + 1] = np.matmul(A1, x_simulated[:, t]) + np.matmul(A2, x_simulated[:, t - 1]) + e_simulated


    #######################################################
    # Constructing matrix Phi such that y = Phi * x, where x = [a1_11, a1_12, a1_21, a1_22, a2_11, a2_12, a2_21, a2_22]:
    print('Constructing matrix Phi...')
    Phi = np.zeros([N * (NT - 3), (N ** 2) * Np])  # Initialising
    for k in range(NT - 3):
        for p in range(Np):
            index = (k + 2) - p
            Phi[2 * k, N * p + N * p: N * (p + 1) + N * p] = x_simulated[:, index]
            Phi[2 * k + 1, N * p + N * (p + 1): N * (p + 1) + N * (p + 1)] = x_simulated[:, index]


    #######################################################
    # Constructing vector y such that y = Phi * x:
    print('Constructing vector y...')
    y = np.zeros([N * (NT - 3)])
    for k in range(NT - 3):
        y[k * N: (k + 1) * N] = x_simulated[:, k + 3]


    #######################################################
    # Estimating A_coef = min_x ||y = Phi * x||, where where x = [a1_11, a1_12, a1_21, a1_22, a2_11, a2_12, a2_21, a2_22]:
    print('Estimating A1 and A2 from simulated time series...')
    Phi_T = np.transpose(Phi)
    inv_Phi_T_Phi = np.linalg.inv(np.matmul(Phi_T, Phi))
    Phi_T_y = np.matmul(Phi_T, y)
    A_estimate = np.matmul(inv_Phi_T_Phi, Phi_T_y)
    A1_estimate = A_estimate.reshape(2, 2, 2)[0]
    A2_estimate = A_estimate.reshape(2, 2, 2)[1]


    #######################################################
    # Printing total computation time:
    computation_time = round(tm.time() - initial_time, 3)  # Initial time stamp
    print('Total computation time:', str(computation_time), 'seconds.')  # Print statement
    basic_tools.print_lines()


    #######################################################
    # Printing results:
    print('A1:         ', A1.reshape(1, 4)), print('A1 estimate:', A1_estimate.reshape(1, 4))
    print('A2:         ', A2.reshape(1, 4)), print('A2 estimate:', A2_estimate.reshape(1, 4))
    basic_tools.print_lines()


    #######################################################
    # Plotting simulated time series:
    print('Plotting...')

    # First element in x_t:
    plt.figure(1)
    plt.plot(x_simulated[0, :])
    plt.title('First element in $x_t$', fontsize=14)
    plt.xlim([0, NT])
    plt.ylim([np.min(x_simulated), np.max(x_simulated)])
    plt.ylabel('First element in $x_t$', fontsize=14)
    plt.xlabel('$t$', fontsize=14)
    plt.grid()

    # Second element in x_t:
    plt.figure(2)
    plt.plot(x_simulated[1, :])
    plt.title('Second element in $x_t$', fontsize=14)
    plt.xlim([0, NT])
    plt.ylim([np.min(x_simulated), np.max(x_simulated)])
    plt.ylabel('Second element in $x_t$', fontsize=14)
    plt.xlabel('$t$', fontsize=14)
    plt.grid()

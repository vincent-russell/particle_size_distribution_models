"""

Title: Estimating VAR(2) coefficients from simulated data (in R^N)
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
    NT = 100000  # Total number of time steps
    sigma_e = np.array([1, 1])  # Standard deviation (in R^2)

    # Case 1:
    # Matrix A1:
    A1 = np.array([[0.3999, 0.1], [0.3, 0.2]])

    # Matrix A2:
    A2 = np.array([[0.0999, 0.4], [0.2, 0.3]])

    # List of coefficient matrices:
    A_list = [A1, A2]

    # Case 2:
    # # Matrix A1:
    # A1 = np.array([[0.3, 0.2, 0.1], [0.25, 0.15, 0.05], [0.12, 0.08, 0.06]])
    #
    # # Matrix A2:
    # A2 = np.array([[0.12, 0.08, 0.06], [0.25, 0.15, 0.05], [0.3, 0.2, 0.1]])
    #
    # # List of coefficient matrices:
    # A_list = [A1, A2]
    #
    # Case 3:
    # Matrix A1:
    # A1 = np.array([[0.25, 0.2, 0.1], [0.25, 0.15, 0.05], [0.12, 0.08, 0.06]])
    #
    # # Matrix A2:
    # A2 = np.array([[0.12, 0.08, 0.06], [0.25, 0.15, 0.05], [0.3, 0.2, 0.1]])
    #
    # # Matrix A3:
    # A3 = np.array([[0.05, 0.05, 0.025], [0.05, 0.02, 0.075], [0.02, 0.08, 0.06]])
    #
    # List of coefficient matrices:
    # A_list = [A1, A2, A3]


    #######################################################
    # Initialising timer for total computation:
    print(), basic_tools.print_lines(), print('Initialising...')
    initial_time = tm.time()  # Initial time stamp


    #######################################################
    # Creating matrix A:
    print('Constructing matrix A...')
    A = np.zeros([Np * N, Np * N])  # Initialising
    for p in range(Np - 1):
        A[0: N, N * p: N * (p + 1)] = A_list[p]
        A[N * (p + 1): N * (p + 2), N * p: N * (p + 1)] = np.eye(N)
    A[0: N, N * (Np - 1): N * Np] = A_list[-1]  # Adding last


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
    x_simulated = np.zeros([N, NT])  # Initialising
    e_simulated = np.zeros(N)  # Initialising
    for t in tqdm(range(1, NT - 1)):
        e_simulated[0: N] = np.random.normal(scale=sigma_e)  # Draw from normal distribution
        x_sum = np.zeros(N)  # Initialising
        for p in range(Np):
            x_sum += np.matmul(A_list[p], x_simulated[:, t - p])
        x_simulated[:, t + 1] = x_sum + e_simulated


    #######################################################
    # Constructing matrix Phi such that y = Phi * x, where x = [a1_11, a1_12, a1_21, a1_22, a2_11, a2_12, a2_21, a2_22]:
    print('Constructing matrix Phi...')
    Phi = np.zeros([N * (NT - Np), (N * N) * Np])  # Initialising
    for k in range(NT - Np):
        for p in range(Np):
            index = (k + (Np - 1)) - p
            for i in range(N):
                Phi[N * k + i, i * N + N * N * p: i * N + N * N * p + N] = x_simulated[:, index]


    #######################################################
    # Constructing vector y such that y = Phi * x:
    print('Constructing vector y...')
    y = np.zeros([N * (NT - Np)])
    for k in range(NT - Np):
        y[k * N: (k + 1) * N] = x_simulated[:, k + Np]


    #######################################################
    # Estimating A_coef = min_x ||y = Phi * x||, where where x = [a1_11, a1_12, a1_21, a1_22, a2_11, a2_12, a2_21, a2_22]:
    print('Estimating A1 and A2 from simulated time series...')
    Phi_T = np.transpose(Phi)
    inv_Phi_T_Phi = np.linalg.inv(np.matmul(Phi_T, Phi))
    Phi_T_y = np.matmul(Phi_T, y)
    A_estimate = np.matmul(inv_Phi_T_Phi, Phi_T_y)
    A_estimate_list = list()
    for p in range(Np):
        A_estimate_list.append(A_estimate.reshape(Np, N, N)[p])


    #######################################################
    # Printing total computation time:
    computation_time = round(tm.time() - initial_time, 3)  # Initial time stamp
    print('Total computation time:', str(computation_time), 'seconds.')  # Print statement
    basic_tools.print_lines()


    #######################################################
    # Printing results:
    for p in range(Np):
        print('A' + str(p + 1) +':         ', A_list[p].reshape(1, N * N)), print('A' + str(p + 1) + ' estimate:', A_estimate_list[p].reshape(1, N * N))
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

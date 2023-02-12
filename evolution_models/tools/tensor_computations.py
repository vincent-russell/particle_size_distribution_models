"""
Matrix and tensor computations
"""


#######################################################
# Modules:
import numpy as np
from scipy.linalg import null_space
from math import floor

# Local modules:
from basic_tools import GLnpt, volume_to_diameter  # ,timer
from evolution_models.tools import get_element_vector, get_element_matrix, Phi_ell_vector


#######################################################
# Computes matrix M:
def compute_M(N, Np, x_boundaries, phi):
    M = np.zeros([N, N])  # Initialising
    for i in range(N):
        ell_i = floor(i / Np)  # ell-th element for phi_i
        degree_i = i - ell_i * Np  # Degree of polynomial i
        for j in range(N):
            ell_j = floor(j / Np)  # ell-th element for phi_j
            degree_j = j - ell_j * Np  # Degree of polynomial j
            # Integrand in M:
            def M_integrand(x):
                return phi[i](x) * phi[j](x)
            GLorder = floor((degree_i + degree_j + 1) / 2) + 2  # Order of integration of Gauss-Legendre quadrature
            M[j, i] = GLnpt(M_integrand, x_boundaries[ell_i], x_boundaries[ell_i + 1], GLorder)  # Computing entries
    return M


#######################################################
# Computes matrix Q:
def compute_Q(cond, N, Np, x_boundaries, phi, dphi, scale_type):
    Q = np.zeros([N, N])  # Initialising
    for i in range(N):
        ell_i = floor(i / Np)  # ell-th element for phi_i
        degree_i = i - ell_i * Np  # Degree of polynomial i
        for j in range(N):
            ell_j = floor(j / Np)  # ell-th element for phi_j
            degree_j = j - ell_j * Np  # Degree of polynomial j
            if ell_i == ell_j:  # Non-zero if in same element
                if scale_type == 'log':  # Checking if using basis based on x = ln(v)
                    # Integrand in Q:
                    def Q_integrand(x):
                        v = np.exp(x)
                        Dp = volume_to_diameter(v)
                        return (3 / Dp) * cond(Dp) * phi[i](x) * dphi[j](x)
                    GLorder = floor((degree_i + degree_j + 1) / 2) + 3  # Order of integration of Gauss-Legendre quadrature
                else:
                    # Integrand in Q:
                    def Q_integrand(v):
                        Dp = volume_to_diameter(v)
                        return (np.pi / 2) * (Dp ** 2) * cond(Dp) * phi[i](v) * dphi[j](v)
                    GLorder = floor((degree_i + degree_j + 1) / 2) + 3  # Order of integration of Gauss-Legendre quadrature
                Q[j, i] = GLnpt(Q_integrand, x_boundaries[ell_i], x_boundaries[ell_i + 1], GLorder)  # Computing entries
    return Q


#######################################################
# Computes matrix R:
# @timer('condensation matrix R')
def compute_R(cond, Ne, Np, N, x_boundaries, phi, inv_M, Q, boundary_zero, scale_type):
    R = np.zeros([N, N])  # Initialising matrix R
    for ell in range(Ne):
        x_ell = x_boundaries[ell]  # x-values of element lower bound
        x_ell_plus_1 = x_boundaries[ell + 1]  # x-values of element upper bound
        Q_ell = get_element_matrix(Q, ell, Np)  # Getting Q_ell matrix
        phi_vector = Phi_ell_vector(phi, ell, Np).get  # Getting Phi_ell(x) vector function
        phi_vector_minus_1 = Phi_ell_vector(phi, max(ell - 1, 0), Np).get  # Getting Phi_{ell-1}(x) vector function
        # Computing R1_ell and R2_ell matrices:
        u1 = np.zeros([Np, 1])  # Initialising vector
        u2 = np.zeros([Np, 1])  # Initialising vector
        u3 = np.zeros([Np, 1])  # Initialising vector
        u1[:, 0] = phi_vector(x_ell_plus_1)  # Vector value
        u2[:, 0] = phi_vector(x_ell)  # Vector value
        u3[:, 0] = phi_vector_minus_1(x_ell)  # Vector value
        if scale_type == 'log':  # Checking if using basis based on x = ln(v)
            Dp_ell = volume_to_diameter(np.exp(x_boundaries[ell]))  # Dp-values of element lower bound
            Dp_ell_plus_1 = volume_to_diameter(np.exp(x_boundaries[ell + 1]))  # Dp-values of element upper bound
            R1_ell = (3 / Dp_ell_plus_1) * cond(Dp_ell_plus_1) * np.matmul(u1, np.transpose(u1))  # R1_ell computation
            R2_ell = (3 / Dp_ell) * cond(Dp_ell) * np.matmul(u2, np.transpose(u3))  # R2_ell computation
        else:
            Dp_ell = volume_to_diameter(x_boundaries[ell])  # Dp-values of element lower bound
            Dp_ell_plus_1 = volume_to_diameter(x_boundaries[ell + 1])  # Dp-values of element upper bound
            R1_ell = (np.pi / 2) * (Dp_ell_plus_1 ** 2) * cond(Dp_ell_plus_1) * np.matmul(u1, np.transpose(u1))  # R1_ell computation
            R2_ell = (np.pi / 2) * (Dp_ell ** 2) * cond(Dp_ell) * np.matmul(u2, np.transpose(u3))  # R2_ell computation
        # Computing tilde_R1_ell and tilde_R2_ell matrices:
        tilde_R1_ell = np.matmul(inv_M[ell], np.subtract(Q_ell, R1_ell))
        tilde_R2_ell = np.matmul(inv_M[ell], R2_ell)
        # Adding elements to matrix:
        if ell == 0:  # First element only adds tilde_R1_ell
            if boundary_zero:  # If boundary condition n(xmin, t) = 0 is imposed:
                R[ell * Np: (ell + 1) * Np, ell * Np: (ell + 1) * Np] = tilde_R1_ell
            else:  # Else boundary condition is not imposed:
                R[ell * Np: (ell + 1) * Np, ell * Np: (ell + 1) * Np] = tilde_R1_ell + tilde_R2_ell
        else:
            R[ell * Np: (ell + 1) * Np, ell * Np: (ell + 1) * Np] = tilde_R1_ell
            R[ell * Np: (ell + 1) * Np, (ell - 1) * Np: ell * Np] = tilde_R2_ell
    return R


#######################################################
# Computes tensor Q_gamma:
# @timer('condensation tensor Q_gamma')
def compute_Q_gamma(Ne, Np, Np_gamma, N_gamma, phi, dphi, phi_gamma, x_boundaries, scale_type):
    Q = np.zeros([Ne, Np, N_gamma, Np])  # Initialising
    # Iterating over elements:
    for ell in range(Ne):
        phi_ell = get_element_vector(phi, ell, Np)
        dphi_ell = get_element_vector(dphi, ell, Np)
        for j in range(Np):
            for i in range(Np):
                for k in range(N_gamma):
                    ell_k = floor(k / Np_gamma)  # ell-th element for phi_k_beta
                    degree_k = k - ell_k * Np_gamma  # Degree of polynomial k
                    if scale_type == 'log':  # Checking if using basis based on x = ln(v)
                        # Integrand in Q_ell:
                        def Q_ell_integrand(x):
                            v = np.exp(x)
                            Dp = volume_to_diameter(v)
                            return (3 / Dp) * phi_gamma[k](np.log(Dp)) * phi_ell[i](x) * dphi_ell[j](x)
                        GLorder = floor((degree_k + i + j + 1) / 2) + 2  # Order of integration of Gauss-Legendre quadrature
                    else:
                        # Integrand in Q_ell:
                        def Q_ell_integrand(v):
                            Dp = volume_to_diameter(v)
                            return (np.pi / 2) * (Dp ** 2) * phi_gamma[k](Dp) * phi_ell[i](v) * dphi_ell[j](v)
                        GLorder = floor((degree_k + i + j + 1) / 2) + 2  # Order of integration of Gauss-Legendre quadrature
                    Q[ell, j, k, i] = GLnpt(Q_ell_integrand, x_boundaries[ell], x_boundaries[ell + 1], GLorder)  # Computing entries
    return Q


#######################################################
# Computes tensor R1_gamma:
def compute_R1_gamma(Ne, Np, N_gamma, phi, phi_gamma, x_boundaries, scale_type):
    R1_gamma = np.zeros([Ne, Np, N_gamma, Np])  # Initialising
    # Iterating over elements:
    for ell in range(Ne):
        phi_ell = get_element_vector(phi, ell, Np)
        x_ell_plus_1 = x_boundaries[ell + 1]
        for j in range(Np):
            for i in range(Np):
                for k in range(N_gamma):
                    if scale_type == 'log':  # Checking if using basis based on x = ln(v)
                        Dp_ell_plus_1 = volume_to_diameter(np.exp(x_ell_plus_1))
                        R1_gamma[ell, j, k, i] = (3 / Dp_ell_plus_1) * phi_gamma[k](np.log(Dp_ell_plus_1)) * phi_ell[i](x_ell_plus_1) * phi_ell[j](x_ell_plus_1)  # Computing entries
                    else:
                        Dp_ell_plus_1 = volume_to_diameter(x_ell_plus_1)
                        R1_gamma[ell, j, k, i] = (np.pi / 2) * (Dp_ell_plus_1 ** 2) * phi_gamma[k](Dp_ell_plus_1) * phi_ell[i](x_ell_plus_1) * phi_ell[j](x_ell_plus_1)  # Computing entries
    return R1_gamma


#######################################################
# Computes tensor R2_gamma:
def compute_R2_gamma(Ne, Np, N_gamma, phi, phi_gamma, x_boundaries, scale_type):
    R2_gamma = np.zeros([Ne, Np, N_gamma, Np])  # Initialising
    # Iterating over elements:
    for ell in range(1, Ne):  # Note: Skips first element.
        phi_ell_minus_1 = get_element_vector(phi, ell - 1, Np)
        phi_ell = get_element_vector(phi, ell, Np)
        x_ell = x_boundaries[ell]
        for j in range(Np):
            for i in range(Np):
                for k in range(N_gamma):
                    if scale_type == 'log':  # Checking if using basis based on x = ln(v)
                        Dp_ell = volume_to_diameter(np.exp(x_ell))
                        R2_gamma[ell, j, k, i] = (3 / Dp_ell) * phi_gamma[k](np.log(Dp_ell)) * phi_ell_minus_1[i](x_ell) * phi_ell[j](x_ell)  # Computing entries
                    else:
                        Dp_ell = volume_to_diameter(x_ell)
                        R2_gamma[ell, j, k, i] = (np.pi / 2) * (Dp_ell ** 2) * phi_gamma[k](Dp_ell) * phi_ell_minus_1[i](x_ell) * phi_ell[j](x_ell)  # Computing entries
    return R2_gamma


#######################################################
# Computes matrix D:
# @timer('deposition matrix D')
def compute_D(depo, N, Np, x_boundaries, phi, scale_type):
    D = np.zeros([N, N])  # Initialising
    for i in range(N):
        ell_i = floor(i / Np)  # ell-th element for phi_i
        degree_i = i - ell_i * Np  # Degree of polynomial i
        for j in range(N):
            ell_j = floor(j / Np)  # ell-th element for phi_j
            degree_j = j - ell_j * Np  # Degree of polynomial j

            # Integrand in D:
            if scale_type == 'log':  # Checking if using basis based on x = ln(v)
                def D_integrand(x):
                    v = np.exp(x)
                    Dp = volume_to_diameter(v)
                    return depo(Dp) * phi[j](x) * phi[i](x)
            else:
                def D_integrand(x):
                    Dp = volume_to_diameter(x)
                    return depo(Dp) * phi[j](x) * phi[i](x)

            if ell_i == ell_j:  # Non-zero if in same element
                GLorder = floor((degree_i + degree_j + 1) / 2) + 2  # Order of integration of Gauss-Legendre quadrature
                D[i, j] = GLnpt(D_integrand, x_boundaries[ell_i], x_boundaries[ell_i + 1], GLorder)  # Computing entries
    return D


#######################################################
# Computes tensor D_eta:
# @timer('deposition tensor D_eta')
def compute_D_eta(N, Np, Np_eta, N_eta, phi, phi_eta, x_boundaries, scale_type):
    D = np.zeros([N, N, N_eta])  # Initialising
    # Iterating over matrices j = 0, 1, ..., N - 1:
    for j in range(N):
        ell_j = floor(j / Np)  # ell-th element for phi_j
        degree_j = j - ell_j * Np  # Degree of polynomial j
        for i in range(N):
            ell_i = floor(i / Np)  # ell-th element for phi_i
            degree_i = i - ell_i * Np  # Degree of polynomial i
            if ell_i == ell_j:  # Non-zero if in same element
                for k in range(N_eta):
                    ell_k = floor(k / Np_eta)  # ell-th element for phi_k_eta
                    degree_k = k - ell_k * Np_eta  # Degree of polynomial k
                    if scale_type == 'log':  # Checking if using basis based on x = ln(v)
                        # Integrand in D^(j)_i,k:
                        def D_integrand(x):
                            v = np.exp(x)
                            Dp = volume_to_diameter(v)
                            return phi[i](x) * phi_eta[k](np.log(Dp)) * phi[j](x)
                    else:
                        # Integrand in D^(j)_i,k:
                        def D_integrand(x):
                            Dp = volume_to_diameter(x)
                            return phi[i](x) * phi_eta[k](Dp) * phi[j](x)
                    GLorder = floor((degree_i + degree_j + degree_k + 1) / 2) + 2  # Order of integration of Gauss-Legendre quadrature
                    D[j, i, k] = GLnpt(D_integrand, x_boundaries[ell_i], x_boundaries[ell_i + 1], GLorder)  # Computing entries
    return D


#######################################################
# Computes matrix A:
def compute_A(N, x_Gauss, phi):
    A = np.zeros([N, N])  # Initialising
    for i in range(N):
        for j in range(N):
            A[i, j] = phi[j](x_Gauss[i])  # Computing entries
    return A


#######################################################
# Computes B and C tensors:
# @timer('coagulation tensors B and C')
def compute_B_C(coag, N, Np, x_boundaries, x_Gauss, phi, scale_type):
    B = np.zeros([N, N, N])  # Initialising
    C = np.zeros([N, N, N])  # Initialising

    # Iterating over i-th matrix in tensor:
    for i in range(N):

        # Determining upper limit of B integral and condition check for non-zero integral from limit value:
        if scale_type == 'log':
            B_var = np.log(np.exp(x_Gauss[i]) - np.exp(x_boundaries[0]))  # Variable upper limit in B_log integral
        else:
            B_var = x_Gauss[i] - x_boundaries[0]  # Variable upper limit in B integral

        # Iterating over k-th entries in i-th matrix:
        for k in range(N):

            # Element and degree of phi_k:
            ell_k = floor(k / Np)  # ell-th element for phi_k
            degree_k = k - ell_k * Np  # Degree of polynomial k

            # Check if B integral will be non-zero due to limit condition:
            B_nonzero_from_lim = True  # Default value
            B_lim = B_var  # Initialising upper limit for B integral
            if x_boundaries[ell_k + 1] < B_var:
                B_lim = x_boundaries[ell_k + 1]
            elif x_boundaries[ell_k] <= B_var <= x_boundaries[ell_k + 1]:
                B_lim = B_var
            else:
                B_nonzero_from_lim = False

            # Iterating over j-th entries in i-th matrix:
            for j in range(N):

                # Element and degree of phi_j:
                ell_j = floor(j / Np)  # ell-th element for phi_j
                degree_j = j - ell_j * Np  # Degree of polynomial j

                # Continuing if limit conditions for B^i_j,k is true:
                if B_nonzero_from_lim:
                # Computing entries for B^i_j,k:
                    # Evaluating integrals and checking if doing log formulation:
                    if scale_type == 'log':
                        # Integrand in B_log:
                        def B_log_integrand(y):
                            xi_y = np.log(abs(np.exp(x_Gauss[i]) - np.exp(y)))
                            cst = 1 / (np.exp(x_Gauss[i]) - np.exp(y))
                            return cst * coag(xi_y, y) * phi[j](xi_y) * phi[k](y)
                        GLorder_B = floor((degree_j + degree_k + 1) / 2) + 3  # Order of integration of Gauss-Legendre quadrature
                        B[i, j, k] = (np.exp(x_Gauss[i]) / 2) * GLnpt(B_log_integrand, x_boundaries[ell_k], B_lim, GLorder_B)
                    else:
                        # Integrand in B:
                        def B_integrand(w):
                            return coag(x_Gauss[i] - w, w) * phi[j](x_Gauss[i] - w) * phi[k](w)
                        GLorder_B = floor((degree_j + degree_k + 1) / 2) + 2  # Order of integration of Gauss-Legendre quadrature
                        B[i, j, k] = (1 / 2) * GLnpt(B_integrand, x_boundaries[ell_k], B_lim, GLorder_B)

                # Computing entries for C^i:
                if phi[j](x_Gauss[i]) != 0:
                    # Evaluating integrals and checking if doing log formulation:
                    if scale_type == 'log':
                        # Integrand in C_log:
                        def C_log_integrand(y):
                            return coag(x_Gauss[i], y) * phi[k](y)
                        GLorder_C = floor((degree_k + 1) / 2) + 3  # Order of integration of Gauss-Legendre quadrature
                        C[i, j, k] = phi[j](x_Gauss[i]) * GLnpt(C_log_integrand, x_boundaries[ell_k], x_boundaries[ell_k + 1], GLorder_C)
                    else:
                        # Integrand in C:
                        def C_integrand(w):
                            return coag(x_Gauss[i], w) * phi[k](w)
                        GLorder_C = floor((degree_k + 1) / 2) + 2  # Order of integration of Gauss-Legendre quadrature
                        C[i, j, k] = phi[j](x_Gauss[i]) * GLnpt(C_integrand, x_boundaries[ell_k], x_boundaries[ell_k + 1], GLorder_C)

    return B, C


#######################################################
# Computes null space U of continuity constraint matrix V such that V * coefficent = 0:
def compute_U(N, Ne, Np, phi, x_boundaries):
    num_constraints = Ne - 1  # Number of continuity constraints
    V = np.zeros([num_constraints, N])  # Initialising
    for c in range(num_constraints):  # Iterating over contraints
        # Left-hand side of boundary:
        for i in range(Np):
            k = c * Np + i  # Constraint * Np + i for i = 0, 1, ..., Np - 1
            V[c, k] = phi[k](x_boundaries[c + 1])  # Computing element
        # Right-hand size of boundary:
        for i in range(Np, 2 * Np):
            k = c * Np + i  # Constraint * Np + i for i = 0, 1, ..., Np - 1
            V[c, k] = -phi[k](x_boundaries[c + 1])  # Computing element
    U = null_space(V)  # Null space of V
    UT = np.transpose(U)  # Transpose of U
    return U, UT


#######################################################
# Computing matrix G for projection operator in BAE computation:
def compute_G(N, N_r, Np, Np_r, phi, phi_r, x_boundaries):
    G = np.zeros([N_r, N])  # Initialising
    for i in range(N):
        ell_i = floor(i / Np)  # ell-th element for phi_i
        degree_i = i - ell_i * Np  # Degree of polynomial i
        for j in range(N_r):
            ell_j = floor(j / Np_r)  # ell-th element for phi^r_j
            degree_j = j - ell_j * Np_r  # Degree of polynomial j

            # Integrand in G:
            def G_integrand(x):
                return phi_r[j](x) * phi[i](x)

            GLorder = floor((degree_i + degree_j + 1) / 2) + 3  # Order of integration of Gauss-Legendre quadrature
            G[j, i] = GLnpt(G_integrand, x_boundaries[ell_i], x_boundaries[ell_i + 1], GLorder)  # Computing entries
    return G

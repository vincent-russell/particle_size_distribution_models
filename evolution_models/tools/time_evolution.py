"""
Time evolution operations
"""

#######################################################
# Modules:
import numpy as np


#######################################################
# Condensation evolution class:
class Condensation_evolution:
    def __init__(self, R):
        self.R = R

    # Evaluation:
    def eval(self, alpha):
        return np.matmul(self.R, alpha)


#######################################################
# Condensation evolution function and class (assuming condensation is unknown):

# Computing matrix R:
def compute_R_unknown(gamma, N, Ne, Np, N_gamma, Q, R1, R2, inv_M, boundary_zero):
    # Precomputations:
    gamma = np.reshape(gamma, (N_gamma, 1))  # Reshape to column vector
    gammaT = np.transpose(gamma)  # Row vector of gamma (transpose)
    # Initialising R:
    R = np.zeros([N, N])
    for ell in range(Ne):
        # Computing tilde_R1_ell:
        gammaT_Q_R1 = np.zeros([Np, Np])
        for j in range(Np):
            Q_R1 = Q[ell][j] - R1[ell][j]
            gammaT_Q_R1[j, :] = np.matmul(gammaT, Q_R1)
        tilde_R1_ell = np.matmul(inv_M[ell], gammaT_Q_R1)
        # Computing tilde_R2_ell:
        gammaT_R2 = np.zeros([Np, Np])
        for j in range(Np):
            gammaT_R2[j, :] = np.matmul(gammaT, R2[ell][j])
        tilde_R2_ell = np.matmul(inv_M[ell], gammaT_R2)
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

# Condensation evolution class:
class Condensation_unknown_evolution:
    def __init__(self, F):
        self.Ne, self.Np, self.N = F.Ne, F.Np, F.N  # Discretisation
        self.N_gamma = F.N_gamma  # Discretisation for gamma
        self.Q = F.Q_gamma  # Tensor Q
        self.R1 = F.R1_gamma  # Tensor R1
        self.R2 = F.R2_gamma  # Tensor R2
        self.inv_M = F.inv_M  # Tensor M_ell^-1
        self.boundary_zero = F.boundary_zero  # Boundary condition n(xmin, t) = 0

    # Evaluation:
    def eval(self, alpha, gamma):
        # Computing matrix R:
        R = compute_R_unknown(gamma, self.N, self.Ne, self.Np, self.N_gamma, self.Q, self.R1, self.R2, self.inv_M, self.boundary_zero)
        # Computing output:
        output = np.matmul(R, alpha)
        return output


#######################################################
# Source evolution class:
class Source_evolution:
    def __init__(self, F, sorc):
        self.sorc = sorc  # Nucleation model
        self.N = F.N  # Discretisation
        self.M = F.M  # Matrix M
        self.phi = F.phi  # Basis functions
        self.xmin = F.x_boundaries[0]  # Lower boundary limit
        self.s = np.zeros(F.N)  # Initialising s(t) vector

    # Evaluation:
    def eval(self, t):
        # Computing vector s(t):
        for i in range(self.N):
            self.s[i] = self.sorc(t) * self.phi[i](self.xmin)
        # Final computation d alpha(t) / dt = output:
        output = np.linalg.solve(self.M, self.s)
        return output


#######################################################
# Source evolution class (assuming source is unknown):
class Source_unknown_evolution:
    def __init__(self, F):
        inv_M = np.linalg.inv(F.M)  # Inverse of matrix M
        Phi_xmin = np.zeros(F.N)  # Initialising vector Phi(xmin)
        for i in range(F.N):
            Phi_xmin[i] = F.phi[i](F.x_boundaries[0])  # Evaluating vector Phi(xmin)
        self.vector = np.matmul(inv_M, Phi_xmin)  # Pre-computed vector

    # Evaluation:
    def eval(self, J):
        # Final computation d alpha(t) / dt = output:
        output = self.vector * J
        return output


#######################################################
# Deposition evolution class:
class Deposition_evolution:
    def __init__(self, F):
        self.N = F.N  # Discretisation
        self.M = F.M  # Matrix M
        self.D = F.D  # Matrix D

    # Evaluation:
    def eval(self, alpha):
        # Computation D(t) x alpha(t):
        b = np.dot(self.D, alpha)
        # Final computation d alpha(t) / dt = output:
        output = -np.linalg.solve(self.M, b)
        return output


#######################################################
# Deposition evolution class (assuming deposition is unknown):
class Deposition_unknown_evolution:
    def __init__(self, F):
        self.N = F.N  # Discretisation
        self.M = F.M  # Matrix M
        self.D_eta = F.D_eta  # Tensor D

    # Evaluation:
    def eval(self, alpha, eta):
        # Quadratic term:
        quad_vec = np.zeros(self.N)
        for j in range(self.N):
            quad_eta = np.dot(self.D_eta[j], eta)
            quad_vec[j] = np.dot(alpha, quad_eta)
        # Final computation d alpha(t) / dt = output:
        output = -np.linalg.solve(self.M, quad_vec)
        return output


#######################################################
# Coagulation evolution class:
class Coagulation_evolution:
    def __init__(self, F):
        self.N = F.N  # Discretisation
        self.A = F.A  # Matrix A
        self.BC = F.B - F.C  # Tensors B and C

    # Evaluation:
    def eval(self, alpha):
        # Quadratic term:
        quad_vec = np.zeros(self.N)
        for i in range(self.N):
            quad_alpha = np.dot(self.BC[i], alpha)
            quad_vec[i] = np.dot(alpha, quad_alpha)
        # Final computation d alpha(t) / dt = output:
        output = np.linalg.solve(self.A, quad_vec)
        return output

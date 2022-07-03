"""
Jacobian of the general dynamic equation evolution operator
"""


#######################################################
# Modules:
import numpy as np

# Local modules:
from basic_tools import Zero
from evolution_models.tools import get_element_vector, compute_R_unknown


#######################################################
# Gets condensation derivative function dF_cond / d_alpha:
def get_dF_cond(F):
    def dF_cond():
        return F.R
    return dF_cond


#######################################################
# Gets condensation derivatives dF_cond / d_alpha and dF_cond / d_gamma (condensation unknown):
def get_dF_cond_unknown(F):

    #######################
    # Precomputation for dF_cond / d_gamma function (computing matrix K):
    def compute_K_unknown(alpha, N, Ne, Np, N_gamma, Q, R1, R2, inv_M, boundary_zero):
        # Initialising K:
        K = np.zeros([N, N_gamma])
        for ell in range(Ne):
            # Obtaining alpha_ell(t) coefficients:
            alpha_ell = get_element_vector(alpha, ell, F.Np)
            alpha_ell_minus_1 = get_element_vector(alpha, max(ell - 1, 0), F.Np)
            # Transposing alpha_ell(t) coefficients:
            alphaT_ell = np.reshape(alpha_ell, (1, F.Np))
            alphaT_ell_minus_1 = np.reshape(alpha_ell_minus_1, (1, F.Np))
            # Computing tilde_K1_ell:
            alphaT_ell_Q_R1_T = np.zeros([Np, N_gamma])
            for j in range(Np):
                Q_R1_T = np.transpose(Q[ell][j] - R1[ell][j])
                alphaT_ell_Q_R1_T[j, :] = np.matmul(alphaT_ell, Q_R1_T)
            tilde_K1_ell = np.matmul(inv_M[ell], alphaT_ell_Q_R1_T)
            # Computing tilde_K2_ell:
            alphaT_ell_minus_1_R2 = np.zeros([Np, N_gamma])
            for j in range(Np):
                alphaT_ell_minus_1_R2[j, :] = np.matmul(alphaT_ell_minus_1, np.transpose(R2[ell][j]))
            tilde_K2_ell = np.matmul(inv_M[ell], alphaT_ell_minus_1_R2)
            # Adding elements to matrix:
            if ell == 0:  # First element only adds tilde_K1_ell
                if boundary_zero:  # If boundary condition n(xmin, t) = 0 is imposed:
                    K[ell * Np: (ell + 1) * Np, :] = tilde_K1_ell
                else:  # Else boundary condition is not imposed:
                    K[ell * Np: (ell + 1) * Np, :] = tilde_K1_ell + tilde_K2_ell
            else:
                K[ell * Np: (ell + 1) * Np, :] = tilde_K1_ell + tilde_K2_ell
        return K

    #######################
    # dF_cond / d_alpha function:
    def dF_cond_alpha(gamma):
        R = compute_R_unknown(gamma, F.N, F.Ne, F.Np, F.N_gamma, F.Q_gamma, F.R1_gamma, F.R2_gamma, F.inv_M, F.boundary_zero)
        return R

    #######################
    # dF_cond / d_gamma function:
    def dF_cond_gamma(alpha):
        K = compute_K_unknown(alpha, F.N, F.Ne, F.Np, F.N_gamma, F.Q_gamma, F.R1_gamma, F.R2_gamma, F.inv_M, F.boundary_zero)
        return K

    # Returning functions dF_cond / d_alpha and dF_cond / d_gamma:
    return dF_cond_alpha, dF_cond_gamma


#######################################################
# Gets deposition derivative function dF_depo / d_alpha:
def get_dF_depo(F):
    #######################
    # Pre-computations:
    dF_depo_output = np.zeros([F.N, F.N])  # Initialising output of dF_depo / d_alpha
    inv_M = np.linalg.inv(F.M)  # Matrix computation
    inv_M_D = np.matmul(inv_M, F.D)  # Matrix computation
    dF_depo_output = -inv_M_D  # Output

    #######################
    # Deposition derivative function dF_depo/ d_alpha:
    def dF_depo():
        return dF_depo_output
    return dF_depo


#######################################################
# Gets deposition derivatives dF_depo / d_alpha and dF_depo / d_eta (deposition unknown):
def get_dF_depo_unknown(F):
    #######################
    # Pre-computations:
    inv_M = np.linalg.inv(F.M)  # Getting M^-1 matrix

    #######################
    # dF_depo / d_alpha function:
    def dF_depo_alpha(eta):
        # Initialisations:
        etaT_DT = np.zeros([F.N, F.N])  # Initialising computation matrix eta^T D^T
        # Pre-loop computations:
        etaT = np.reshape(eta, (1, F.N_eta))  # Transpose of eta
        for j in range(F.N):
            # Computing eta^T D^T:
            etaT_DT[j, :] = np.matmul(etaT, np.transpose(F.D_eta[j]))
        # Final output computation:
        output = -np.matmul(inv_M, etaT_DT)
        return output

    #######################
    # dF_depo / d_eta function:
    def dF_depo_eta(alpha):
        # Initialisations:
        alphaT_D = np.zeros([F.N, F.N_eta])  # Initialising computation matrix alpha^T D
        # Pre-loop computations:
        alphaT = np.reshape(alpha, (1, F.N))  # Transpose of alpha
        for j in range(F.N):
            # Computing alpha^T D:
            alphaT_D[j, :] = np.matmul(alphaT, F.D_eta[j])
        # Final output computation:
        output = -np.matmul(inv_M, alphaT_D)
        return output

    # Returning functions dF_depo / d_alpha and dF_depo / d_eta:
    return dF_depo_alpha, dF_depo_eta


#######################################################
# Gets source derivative function dF_sorc / d_alpha:
def get_dF_sorc(F):
    #######################
    # Pre-computations:
    dF_sorc_output = np.zeros([F.N, F.N])  # Output

    #######################
    # Source derivative function dF_sorc/ d_alpha:
    def dF_sorc():
        return dF_sorc_output
    return dF_sorc


#######################################################
# Gets source derivative dF_sorc / d_J (source unknown):
def get_dF_sorc_unknown(F):
    #######################
    # Pre-computations:
    inv_M = np.linalg.inv(F.M)  # Getting M^-1 matrix
    Phi_xmin = np.zeros([F.N, 1])  # Initialising vector Phi(xmin)
    for i in range(F.N):
        Phi_xmin[i] = F.phi[i](F.x_boundaries[0])  # Evaluating vector Phi(xmin)
    dF_sorc_output = np.matmul(inv_M, Phi_xmin)  # Output

    #######################
    # Source derivative function dF_sorc/ d_alpha:
    def dF_sorc():
        return dF_sorc_output

    return dF_sorc


#######################################################
# Gets coagulation derivative function dF_coag / d_alpha:
def get_dF_coag(F):
    #######################
    # Pre-computations:
    BC = F.B - F.C  # Tensor subtraction operation
    BCT = np.zeros([F.N, F.N, F.N])  # Initialising tensor (B - C) + (B - C)^T
    for i in range(F.N):
        BCT[i] = BC[i] + np.transpose(BC[i])  # Computing tensor (B - C) + (B - C)^T

    #######################
    # Coagulation derivative function dF_coag / d_alpha:
    def dF_coag(alpha):
        output = np.zeros([F.N, F.N])  # Initialising output
        for i in range(F.N):
            output[i, :] = np.matmul(np.transpose(alpha), BCT[i])  # Computing output
        return output
    return dF_coag


#######################################################
# Jacobian of general dynamic equation evolution model class:
class GDE_Jacobian:
    def __init__(self, F):
        # Print statement:
        if F.print_status:
            print('Initialising Jacobian of general dynamic equation evolution operator.')
        # Properties for evaluation:
        self.N = F.N  # Total degrees of freedom
        self.dt = F.dt  # Time step
        self.unknowns = F.unknowns  # Saving unknowns
        self.time_integrator = F.time_integrator  # Saving time integrator method
        # Initialising derivatives (as zero vector functions):
        self.dF_cond = Zero(N=F.N, M=F.N).function_array  # Condensation derivative function initialisation
        self.dF_sorc = Zero(N=F.N, M=F.N).function_array  # Source derivative function initialisation
        self.dF_depo = Zero(N=F.N, M=F.N).function_array  # Deposition derivative function initialisation
        self.dF_coag = Zero(N=F.N, M=F.N).function_array  # Coagulation derivative function initialisation
        # Initialising derivaties of unknowns (as zero vector functions):
        self.dF_cond_alpha = Zero(N=F.N, M=F.N).function_array  # Condensation derivative function alpha initialisation
        self.dF_cond_gamma = Zero(N=F.N, M=F.N_gamma).function_array  # Condensation derivative function gamma initialisation
        self.dF_depo_alpha = Zero(N=F.N, M=F.N).function_array  # Deposition derivative function alpha initialisation
        self.dF_depo_eta = Zero(N=F.N, M=F.N_eta).function_array  # Deposition derivative function eta initialisation
        # Computing derivatives (and of unknowns):
        if 'condensation' in F.unknowns:
            self.N_gamma = F.N_gamma  # Total degrees of freedom for gamma
            self.dF_cond_alpha, self.dF_cond_gamma = get_dF_cond_unknown(F)  # Condensation (unknown) derivative function computation
        elif F.f_cond is not Zero.function:
            self.dF_cond = get_dF_cond(F)  # Condensation derivative function computation
        if 'deposition' in F.unknowns:
            self.N_eta = F.N_eta  # Total degrees of freedom for eta
            self.dF_depo_alpha, self.dF_depo_eta = get_dF_depo_unknown(F)  # Deposition (unknown) derivative function computation
        elif F.f_depo is not Zero.function:
            self.dF_depo = get_dF_depo(F)  # Deposition derivative function computation
        if 'source' in F.unknowns:
            self.dF_sorc_J = get_dF_sorc_unknown(F)  # Source (unknown) derivative function computation
        elif F.f_sorc is not Zero.function:
            self.dF_sorc = get_dF_sorc(F)  # Source derivative function computation
        if F.f_coag is not Zero.function:
            self.dF_coag = get_dF_coag(F)  # Coagulation derivative function computation

    # Jacobian evaluation dF_alpha / d_alpha:
    def eval_d_alpha(self, x, *_):
        if 'condensation' in self.unknowns and 'deposition' in self.unknowns:
            alpha = x[0: self.N]
            gamma = x[self.N: self.N + self.N_gamma]
            eta = x[self.N + self.N_gamma: self.N + self.N_gamma + self.N_eta]
            if self.time_integrator == 'euler':
                return np.eye(self.N) + self.dt * (self.dF_cond_alpha(gamma) + self.dF_sorc() + self.dF_depo_alpha(eta) + self.dF_coag(alpha))
            else:
                return self.dF_cond_alpha(gamma) + self.dF_sorc() + self.dF_depo_alpha(eta) + self.dF_coag(alpha)
        elif 'condensation' in self.unknowns:
            alpha = x[0: self.N]
            gamma = x[self.N: self.N + self.N_gamma]
            if self.time_integrator == 'euler':
                return np.eye(self.N) + self.dt * (self.dF_cond_alpha(gamma) + self.dF_sorc() + self.dF_depo() + self.dF_coag(alpha))
            else:
                return self.dF_cond_alpha(gamma) + self.dF_sorc() + self.dF_depo() + self.dF_coag(alpha)
        elif 'deposition' in self.unknowns:
            alpha = x[0: self.N]
            eta = x[self.N: self.N + self.N_eta]
            if self.time_integrator == 'euler':
                return np.eye(self.N) + self.dt * (self.dF_cond() + self.dF_sorc() + self.dF_depo_alpha(eta) + self.dF_coag(alpha))
            else:
                return self.dF_cond() + self.dF_sorc() + self.dF_depo_alpha(eta) + self.dF_coag(alpha)
        else:
            alpha = x[0: self.N]
            if self.time_integrator == 'euler':
                return np.eye(self.N) + self.dt * (self.dF_cond() + self.dF_sorc() + self.dF_depo() + self.dF_coag(alpha))
            else:
                return self.dF_cond() + self.dF_sorc() + self.dF_depo() + self.dF_coag(alpha)

    # Jacobian evaluation dF_alpha / d_gamma:
    def eval_d_gamma(self, x, *_):
        alpha = x[0: self.N]
        if self.time_integrator == 'euler':
            return self.dt * self.dF_cond_gamma(alpha)
        else:
            return self.dF_cond_gamma(alpha)

    # Jacobian evaluation dF_alpha / d_eta:
    def eval_d_eta(self, x, *_):
        alpha = x[0: self.N]
        if self.time_integrator == 'euler':
            return self.dt * self.dF_depo_eta(alpha)
        else:
            return self.dF_depo_eta(alpha)

    # Jacobian evaluation dF_alpha / d_J:
    def eval_d_J(self, *_):
        if self.time_integrator == 'euler':
            return self.dt * self.dF_sorc_J()
        else:
            return self.dF_sorc_J()

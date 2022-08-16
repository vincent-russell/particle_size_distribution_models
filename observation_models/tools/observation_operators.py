"""
Particle size distribution observation operator
"""


#######################################################
# Modules:
import numpy as np


#######################################################
# Particle size distribution observation model class:
class Size_distribution_observation_model:

    def __init__(self, F, d_obs, M):
        # Print statement:
        print('Computing size distribution observation operator...')
        # Parameters:
        v_obs = (1 / 6) * np.pi * (d_obs ** 3)  # Diameter to volume
        constant = np.zeros(M)  # Initialising constant
        if F.scale_type == 'log':
            x_obs = np.log(v_obs)  # Log of sizes where observations are made
            for i in range(M):
                constant[i] = 3 / np.log10(np.e)  # Constant evaluation
        else:
            x_obs = v_obs  # Sizes where observations are made
            for i in range(M):
                constant[i] = (np.pi / 2) * (d_obs[i] ** 2)  # Constant evaluation
        # Computing H_phi matrix:
        self.H_phi = np.zeros([M, F.N])  # Initialising observation matrix H_phi
        for j in range(F.N):
            for i in range(M):
                self.H_phi[i, j] = constant[i] * F.phi[j](x_obs[i])  # Computing elements of matrix H_phi

    # Observation operator function H = H(alpha, t)
    def eval(self, alpha, *_):
        return np.matmul(self.H_phi, alpha)

"""
Jacobian of the particle size distribution observation operator
"""


#######################################################
# Jacobian of particle size distribution observation model class:
class Size_distribution_observation_model_Jacobian:

    def __init__(self, H):
        # Print statement:
        print('Computing Jacobian of size distribution observation operator...')
        # Parameters:
        self.H_phi = H.H_phi

    # Jacobian evaluation:
    def eval(self, *_):
        return self.H_phi

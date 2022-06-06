"""
Functions associated with random noise processes
"""


#######################################################
# Modules:
import numpy as np
import numpy.random as rand

# Local modules:
from basic_tools.miscellaneous import positive


#######################################################
# Transforms 1D or 2D array n with dimensions N or (Nx, Ny) to poisson distributed array:
def get_poisson(n):
    if np.ndim(n) == 1:  # If 1-D array
        N = np.shape(n)[0]  # Dimension
        n_pois = np.zeros(N)  # Initialising
        for i in range(N):
            n_pois[i] = rand.poisson(positive(n[i]), 1)[0]
    elif np.ndim(n) == 2:  # If 2-D array
        Nx, Ny = np.shape(n)  # Dimensions
        n_pois = np.zeros([Nx, Ny])  # Initialising
        for i in range(Nx):
            for j in range(Ny):
                n_pois[i, j] = rand.poisson(positive(n[i, j]), 1)[0]
    else:
        print('Error in get_poisson(n): not 1D or 2D numpy array. returned input array n')
        n_pois = n
    return n_pois

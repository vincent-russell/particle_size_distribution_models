"""
Evaluation functions
"""


#######################################################
# Modules:
import numpy as np

# Local modules:
from basic_tools import get_kwarg_value


#######################################################
# Function to compute norm difference between two arrays through time:
def compute_norm_difference(n_truth, n_estimate, *sigma_n, **kwargs):
    # Parameters:
    compute_weighted_norm = get_kwarg_value(kwargs, 'compute_weighted_norm', False)  # Set to True to compute weighted norm difference using variance (sigma_n)
    print_statements = get_kwarg_value(kwargs, 'print_statements', True)  # Set to False to disable print statements
    print_rounding = get_kwarg_value(kwargs, 'print_rounding', 0)  # Option to control print round value
    # Computation:
    N, NT = np.shape(n_truth)  # Dimensions
    n_diff = n_estimate - n_truth  # Computing difference between estimate and truth
    norm_diff = np.zeros(NT)  # Initialising norm difference for each time step
    if compute_weighted_norm:  # Set to True to compute norm difference weighted by variance
        sigma_n = sigma_n[0] + 1e-5  # Adding small number to sigma (to make non-singular)
        for k in range(NT):  # Iterating over time
            norm_diff[k] = np.sqrt(np.matmul(n_diff[:, k], np.matmul(np.diag(1 / (sigma_n[:, k] ** 2)), n_diff[:, k])))  # Computing norm
        if print_statements:
            print('Total weighted norm difference between estimate and truth:', str(round(np.linalg.norm(norm_diff), print_rounding)))
    else:
        for k in range(NT):  # Iterating over time
            norm_diff[k] = np.sqrt(np.matmul(n_diff[:, k], n_diff[:, k]))  # Computing norm
        if print_statements:
            print('Total norm difference between estimate and truth:', str(round(np.linalg.norm(norm_diff), print_rounding)))
    return norm_diff

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
    is_1D = get_kwarg_value(kwargs, 'is_1D', False)  # Set to True to compute norm for 1D array
    compute_weighted_norm = get_kwarg_value(kwargs, 'compute_weighted_norm', False)  # Set to True to compute weighted norm difference using variance (sigma_n)
    print_statements = get_kwarg_value(kwargs, 'print_statements', True)  # Set to False to disable print statements
    print_rounding = get_kwarg_value(kwargs, 'print_rounding', 0)  # Option to control print round value
    print_name = get_kwarg_value(kwargs, 'print_name', None)  # Set print name
    # Pre-computation:
    if is_1D:
        NT = len(n_truth)  # Dimensions
    else:
        N, NT = np.shape(n_truth)  # Dimensions
    n_diff = n_estimate - n_truth  # Computing difference between estimate and truth
    norm_diff = np.zeros(NT)  # Initialising norm difference for each time step
    # Norm computation:
    if compute_weighted_norm:  # Set to True to compute norm difference weighted by variance
        sigma_n = sigma_n[0] + 1e-5  # Adding small number to sigma (to make non-singular)
        for k in range(NT):  # Iterating over time
            if is_1D:
                norm_diff[k] = abs(n_diff[k]) / sigma_n[k]  # Computing norm
            else:
                norm_diff[k] = np.sqrt(np.matmul(n_diff[:, k], np.matmul(np.diag(1 / (sigma_n[:, k] ** 2)), n_diff[:, k])))  # Computing norm
    else:
        for k in range(NT):  # Iterating over time
            if is_1D:
                norm_diff[k] = abs(n_diff[k])  # Computing norm
            else:
                norm_diff[k] = np.sqrt(np.matmul(n_diff[:, k], n_diff[:, k]))  # Computing norm
    # Print statements:
    if print_statements:
        if print_name is None:
            print('Total norm difference between estimate and truth:', str(round(np.linalg.norm(norm_diff), print_rounding)))
        else:
            print('Total norm difference of ' + print_name + ' estimate and truth: ', str(round(np.linalg.norm(norm_diff), print_rounding)))
    return norm_diff

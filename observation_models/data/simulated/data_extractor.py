"""
Extracts data from simulated observations
"""


#######################################################
# Modules:
import os
from numpy import load


#######################################################
# Load simulated observations:
def load_observations(data_filename):
    data_path = os.path.join(os.path.dirname(__file__), data_filename + '.npz')  # Gets path to data file
    return load(data_path)

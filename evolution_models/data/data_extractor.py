"""
Extracts data from computed coagulation tensors
"""


#######################################################
# Modules:
import os
from numpy import load


#######################################################
# Load coagulation tensors:
def load_coagulation_tensors(data_filename):
    data_path = os.path.join(os.path.dirname(__file__), data_filename + '.npz')  # Gets path to data file
    return load(data_path)

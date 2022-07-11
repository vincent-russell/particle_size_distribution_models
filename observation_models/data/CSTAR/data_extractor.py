"""

Title: Extracts data from CSTAR data.csv file
Author: Vincent Russell
Date: June 6, 2020

"""


#######################################################
# Modules:
import os
import numpy as np
import pandas as pd


#######################################################
# Get size Dp and time from .csv data file:
def get_Dp_time(array):

    # Get size Dp (nm?):
    Dp = array[28:135, 0]  # Extract Dp from data array
    Dp = Dp.astype(np.float)  # Convert strings in array to floats

    # Get time (hours):
    time_string = array[26, 25:]  # Time (hh:mm:ss)

    # Function to convert hh:mm:ss to hours:
    def get_sec(time_str):
        h, m, s = time_str.split(':')
        return int(h) + int(m) / 60 + int(s) / 3600

    # Convert time to hours:
    NT = len(time_string)  # Number of time steps
    time = np.zeros(NT)  # Initialising time vector
    for i in range(NT):
        time[i] = get_sec(time_string[i])  # Converting time to hours
    time = time - time[0]  # Shift time to start from zero

    return Dp, time


#######################################################
# Extracting data from .csv file:
def get_CSTAR_data():
    data_path = os.path.join(os.path.dirname(__file__), "data.csv")  # Gets path to data file
    array = pd.read_csv(data_path).to_numpy()  # Extracts data from .csv file into numpy array
    Dp, time = get_Dp_time(array)  # Size Dp and time (hours)
    n = array[28:135, 25:].astype(np.float)  # Size distribution data
    Dp = Dp / 1000  # Coverting Dp from nanometers to micrometers
    return Dp, n, time

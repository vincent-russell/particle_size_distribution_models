"""

Title: Extracts data from piera data.csv file
Author: Vincent Russell
Date: July 7, 2022

"""


#######################################################
# Modules:
import os
import numpy as np
import pandas as pd


#######################################################
# Extracting Piera data from .csv file:
def get_piera_data(source_name, event_number, serial_number):
    """
    Returns Piera data given the source name, event number, and serial number

            Parameters:
                    source_name (str) : Source of aerosol which is Axe, Vape, or Tobacco
                    event_number (int) : Event identifier which for Axe is 0-2, Vape is 3-7, and Tobacco is 8-10
                    serial_number (int) : Serial number of sensor which can be 1-10

            Returns:
                    Dp (1D array) : Array of diameter of aerosol sizes (micro meters)
                    Y (2D array) : Particle count data
                    time (1D array) : Time (in minutes) that observations are made

    """

    # Array of diameter of aerosol sizes (micro meters):
    Dp = np.array([0.1, 0.3, 0.5, 1.0, 2.5, 5.0, 10.0])
    # Path to data:
    data_path = os.path.dirname(__file__) + '/data_clean/' + source_name + '/' + '21-03-2022' + '-serial-' + str(serial_number) + '-event-' + str(event_number) + '.csv'
    # Extracts data from .csv file into numpy array:
    array = pd.read_csv(data_path).to_numpy()
    # Separating time and particle count data:
    time = array[:, 0]  # Time data
    Y = array[:, 1:]  # Particle count data
    return Dp, Y, time

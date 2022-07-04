"""

Title: Cuts and Cleans Piera Systems aerosol data
Author: Vincent Russell
Date: June 22, 2022

"""


#######################################################
# Modules:
from os import mkdir
import numpy as np
import time as tm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

# Local modules:
import basic_tools


#######################################################
if __name__ == '__main__':

    #######################################################
    # Parameters:
    plot_aerosol_1D = False  # Set to true to plot aerosol in 1D
    event_number = 5  # Event number (Deodorant: 0-2, Vape: 3-7, Smoke: 8-10)
    size_number = 0  # Particle size number
    plot_aerosol_2D = False  # Plot heat map of aerosol particle counts
    plot_aerosol_2D_log = True  # Set to true if plotting log(aerosol particle counts)
    aerosol_N_time_max = 68  # Maximum number of time steps allowed in aerosol count data for each event
    save_data = True  # Set to true to save dataset
    data_create_dir = False  # Set to true to also create directory for saved data
    data_include_time = True  # Set to true to also include time in data .csv files
    event_type = 'Vape'  # Axe, Vape, Tobacco, or All
    data_load_from_path = 'C:/Users/Vincent/OneDrive - The University of Auckland/Python/particle_size_distribution_models/observation_models/data/piera/data_raw/'  # Path of data to load from
    data_save_to_path = 'C:/Users/Vincent/OneDrive - The University of Auckland/Python/particle_size_distribution_models/observation_models/data/piera/data_clean/'  # Path of data to save to


    #######################################################
    # Initialising timer for total computation:
    basic_tools.print_lines()
    initial_time = tm.time()  # Initial time stamp


    #######################################################
    # Pre-loop initialisations:
    aerosol_sizes = np.array([0.1, 0.3, 0.5, 1.0, 2.5, 5.0, 10])  # Array of aerosol sizes (diameter in micro meters)
    event_observations = list()  # Initialising list of observations for each event
    event_observations_log = list()  # Initialising list of log observations for each event
    event_time = list()  # Initialising list of times for each event
    event_source = list()  # Initliainsg list of source for each event


    #######################################################
    # Pre-loop initialisations:
    if data_create_dir:
        mkdir(data_save_to_path + event_type + '/')


    #######################################################
    # Iterating over serial number:
    print('Iterating over serial number ' + event_type)
    for serial_number in tqdm(np.arange(1, 11)):


        #######################################################
        # Loading dataframe:
        event_df = pd.read_csv(data_load_from_path + 'events.csv')
        round_1_df = pd.read_csv(data_load_from_path + 'round_1/Serial_' + str(serial_number) + '.csv')
        round_2_df = pd.read_csv(data_load_from_path + 'round_2/Serial_' + str(serial_number) + '.csv')
        dataframe = pd.concat([round_1_df, round_2_df])


        #######################################################
        # Cleaning time data:

        # Event time:
        event_df = event_df.drop(16)  # Removing repeated row for 'source' activity
        event_time_start_raw = pd.to_datetime(event_df[event_df['Activity'] == 'Source']['Date'] + 'T' + event_df[event_df['Activity'] == 'Source']['Time']).to_numpy()  # Getting datetime for 'source' activity
        event_time_stop_raw = pd.to_datetime(event_df[event_df['Activity'] == 'Ventilate']['Date'] + 'T' + event_df[event_df['Activity'] == 'Ventilate']['Time']).to_numpy()  # Getting datetime for 'source' activity
        initial_time_raw = event_time_start_raw[0]

        # Time in aerosol data:
        dataframe['Time'] = dataframe['Time'].str.replace('PDT', '')  # Removing timezone in dataframe
        time_raw = pd.to_datetime(dataframe['Time']).to_numpy()  # Raw datetime data as array

        # Converting time to minutes:
        event_time_start = (event_time_start_raw - initial_time_raw) / np.timedelta64(1, 'm')
        event_time_stop = (event_time_stop_raw - initial_time_raw) / np.timedelta64(1, 'm')
        time = (time_raw - initial_time_raw) / np.timedelta64(1, 'm')


        #######################################################
        # Converting aerosol data to array:
        observations = dataframe.iloc[:, 2:9].to_numpy()  # Aerosol particle counts
        observations_log = np.log(observations + 1)  # Log transformation of particle counts


        #######################################################
        # Cutting aerosol data into each event:
        N_sizes = len(aerosol_sizes)  # Total number of sizes
        N_time = len(time)  # Total number of time steps
        N_events = len(event_time_start)  # Total number of events

        event_observations = list()  # Initialising list of observations for each event
        event_observations_log = list()  # Initialising list of log observations for each event
        event_time = list()  # Initialising list of times for each event

        for i in range(N_events):  # Iterating over number of events
            event_i_obs = np.empty(shape = (0, 7))  # Initialising array of i-th event observations
            event_i_obs_log = np.empty(shape = (0, 7))  # Initialising array of i-th event log observations
            event_i_time = np.array([])  # Initialising array of i-th event time
            for k in range(N_time):  # Iterating over time indices
                if event_time_start[i] < time[k] < event_time_stop[i]:  # If the time value is within the event time
                    event_i_time = np.append(event_i_time, time[k])  # Appending i-th event time
                    event_i_obs = np.append(event_i_obs, np.reshape(observations[k, :], (1, 7)), axis = 0)  # Appending i-th event aerosol counts
                    event_i_obs_log = np.append(event_i_obs_log, np.reshape(observations_log[k, :], (1, 7)), axis = 0)  # Appending i-th event log aerosol counts
            # Cutting observations to maximum time size:
            event_i_obs = event_i_obs[0: aerosol_N_time_max]
            event_i_obs_log = event_i_obs_log[0: aerosol_N_time_max]
            # Cutting time to maximum time size and zero-ing time (setting first element to 0):
            event_i_time = event_i_time[0: aerosol_N_time_max]
            event_i_time = event_i_time - event_i_time[0]
            # Adding i-th event data to lists and cutting to max time index:
            event_observations.append(event_i_obs)  # Adding i-th event observations to list
            event_observations_log.append(event_i_obs_log)  # Adding i-th event log observations to list
            event_time.append(event_i_time)  # Adding i-th event times to list
        event_source = event_df[event_df['Activity'] == 'Source']['Source'].to_numpy()  # List of aerosol source for each event


        #######################################################
        # Getting indices for specified aerosol type:
        number_events = list(event_source).count(event_type)  # Number of events of aerosol event type
        if event_type == 'Axe':
            event_indices = [0, 1, 2]
        elif event_type == 'Vape':
            event_indices = [3, 4, 5, 6, 7]
        elif event_type == 'Tobacco':
            event_indices = [8, 9, 10]
        else:
            event_indices = np.arange(N_events)


        #######################################################
        # Saving data to .csv file:
        if save_data:
            date_string = '21-03-2022'  # String of date of data
            for i in event_indices:  # Iterating over number of events
                if data_include_time:
                    data_header = 'Time (mins), PC0.1, PC0.3, PC0.5, PC1.0, PC2.5, PC5.0, PC10'  # Header for .csv file for data
                    # Appending time array to observation array if including time in saved data:
                    event_observations_and_time = np.append(np.reshape(event_time[i], (-1, 1)), event_observations[i], axis=1)
                    # Saving data of i-th event in .csv file:
                    with open(data_save_to_path + event_type + '/' + date_string + '-serial-' + str(serial_number) + '-event-' + str(i) + '.csv', 'a') as csvfile:
                        np.savetxt(csvfile, event_observations_and_time, delimiter = ',', header = data_header, fmt = '%s', comments = '')
                else:
                    data_header = 'PC0.1, PC0.3, PC0.5, PC1.0, PC2.5, PC5.0, PC10'  # Header for .csv file for data
                    # Saving data of i-th event in .csv file:
                    with open(data_save_to_path + event_type + '/' + date_string + '-serial-' + str(serial_number) + '-event-' + str(i) + '.csv', 'a') as csvfile:
                        np.savetxt(csvfile, event_observations[i], delimiter = ',', header = data_header, fmt = '%s', comments = '')


    #######################################################
    # Plotting aerosol observations:
    if plot_aerosol_2D:
        yticks = aerosol_sizes.tolist()
        plt.figure(2)
        if plot_aerosol_2D_log:
            plt.pcolor(event_time[event_number], aerosol_sizes, np.transpose(event_observations[event_number]), cmap = 'jet', norm = LogNorm())
            plt.colorbar(label = 'Particle counts per litre of air')
        else:
            plt.pcolor(event_time[event_number], aerosol_sizes, np.transpose(event_observations[event_number]), cmap = 'jet')
            plt.colorbar(label = 'Particle counts per litre of air')
        plt.clim(1, 1e7)
        plt.title('Particle counts per litre from ' + event_source[event_number])
        plt.xlabel('Time (minutes)')
        plt.ylabel('Particle Diameter ($\mu$m)')
        plt.xlim(min(event_time[event_number]), max(event_time[event_number]))
        plt.yscale('log')
        plt.ylim(aerosol_sizes[0], aerosol_sizes[-1])
        plt.yticks(yticks, [str(x) for x in yticks])

    # Final print statements
    basic_tools.print_lines()  # Print lines in console
    print()  # Print space in console

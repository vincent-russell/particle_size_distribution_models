"""
File saving and loading with date and time in filename
"""


#######################################################
# Modules:
from numpy import savez, load
from datetime import datetime
from os import path, listdir, remove, stat

# Local modules:
from basic_tools.miscellaneous import get_array_name


#######################################################
# Save data in file (directory saved_data) with current date and time in filename,
# and deletes oldest data. Has keyword argument to change age to delete oldest data:
def save_data(namespace, *datafiles, **kwargs):
    # Current date and time:
    now = datetime.now()

    # Function to convert digits to string, and add 0 if only has 1 digit:
    def string_check_add_zero(digits):
        string_digits = str(digits)  # Converting to string
        if len(string_digits) == 1:  # Check if has only 1 digit
            string_digits = '0' + string_digits  # Adds 0 to digit
        return string_digits

    # Converting date and time to strings, and check if need to add 0:
    year, month, day = string_check_add_zero(now.year), string_check_add_zero(now.month), string_check_add_zero(now.day)
    hour, minute, second = string_check_add_zero(now.hour), string_check_add_zero(now.minute), string_check_add_zero(now.second)

    # Concatenating date and time strings:
    date_and_time = year + '-' + month + '-' + day + '_' + hour + '.' + minute + '.' + second

    # Joining path to directory 'saved_data\' with date and time strings:
    pathname = path.join('saved_data', date_and_time)

    # Concatenating filename (and parameter.py string) with 'saved_data\' path date and time strings:
    path_filename = pathname + '_' + 'datafiles'
    path_parameters = pathname + '_' + 'parameters'

    # Creating dictionary of array names and array data (to pass savez **kwargs):
    data_dictionary = {}  # Initialising dictionary
    for datafile in datafiles:  # Iterating over datafiles
        datafile_name = get_array_name(datafile, namespace)  # Gets array name as string
        data_dictionary[datafile_name] = datafile  # Adds array to dictionary

    # Deleting old data in 'saved_data\' if exceeds 'age_delete' (deletes max 2 data):
    if 'age_delete' in kwargs:
        age_delete = kwargs['age_delete']
    else:
        age_delete = 30 * 2  # Default maximum number of data in folder after being deleted
    if len(listdir('saved_data')) >= age_delete:  # Condition to delete oldest data
        name_list, age_list = age_sorted_filename_lists('saved_data')
        remove(name_list[0])  # Removes oldest file
        remove(name_list[1])  # Removes second oldest file

    # Saving data in .npz file, and parameters in .txt file:
    savez(path_filename, **data_dictionary)
    copy_code('main', path_parameters)

    # Final print statement:
    print('Saved solution data, basis functions, and evolution operators.')


#######################################################
# Returns two lists, one of the names of the data in subdirectory, and the other
# of the age of the filenames (sorted from oldest to youngest):
def age_sorted_filename_lists(subdirectory):
    name_list = []  # Initialising list of names of data
    age_list = []  # Initialising list of age of data
    for file in listdir(subdirectory):  # Iterating over filenames
        file = path.join(subdirectory, file)  # Adding path to filenames
        name_list.append(file)  # Adding filename to name_list
        age_list.append(stat(file).st_mtime)  # Adding age of file to list
    name_list, age_list = zip(*sorted(zip(name_list, age_list)))  # Simultaneous sorting of data
    return name_list, age_list


#######################################################
# Copy contents of .py file as .txt file:
def copy_code(python_filename, output_filename):
    with open(python_filename + '.py') as file1:
        with open(output_filename + '.txt', 'w') as file2:
            for line in file1:  # Iterating over each line in .py file
                file2.write(line)  # Writing each line in .txt file


#######################################################
# Load data in file with date and time in filename, format 'YYYY-MM=DD_HH.MM.SS',
# with option to return specific data given their name:
def load_data(datetime, *datanames):
    # Concatenating date and time strings to filename:
    filename = datetime + '_' + 'datafiles' + '.npz'

    # Joining path to directory 'saved_data\' with filename:
    pathname = path.join('saved_data', filename)

    # Loading datafile:
    datafile = load(pathname, allow_pickle=True)  # Allow_pickle true for condition 'datanames is ()'

    # Default: If no names are given, all data is extracted in datafile:
    if datanames is ():
        datanames = datafile.files

    # Iterate over datanames to extract from datafile:
    if len(datanames) > 1:  # If extracting more than one file, returns list
        data_list = []
        for dataname in datanames:
            data_array = datafile[dataname]
            data_list.append(data_array)
    else:
        data_list = datafile[datanames[0]]  # If extracting only one file, returns array

    return data_list  # Returning list (or single array) of data


#######################################################
# Load most recent datafile in folder 'saved_data\',
# with option to return specific data given their name:
def load_most_recent(*datanames):

    # Getting sorted list of filename and age of data:
    name_list, age_list = age_sorted_filename_lists('saved_data')

    # Concatenating name of most recent file with .npz:
    filename = name_list[-2]

    # Joining path to directory 'saved_data\' with filename:
    # pathname = path.join('saved_data', filename)

    # Loading datafile:
    datafile = load(filename, allow_pickle=True)  # Allow_pickle true for condition 'datanames is ()'

    # Default: If no names are given, all data is extracted in datafile:
    if datanames is ():
        datanames = datafile.files

    # Iterate over datanames to extract from datafile:
    if len(datanames) > 1:
        data_list = []
        for dataname in datanames:
            data_array = datafile[dataname]
            data_list.append(data_array)
    else:
        data_list = datafile[datanames[0]]

    return data_list  # Returning list of data

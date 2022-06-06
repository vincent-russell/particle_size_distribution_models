"""
File saving and loading using csv
"""


#######################################################
# Modules:
from numpy import size, round
from csv import writer

# Local modules:
from basic_tools.miscellaneous import get_kwarg_value


#######################################################
# Function to save 1D or 2D numpy arrays to csv:
def save_array_to_csv(filename, array, **kwargs):
    row_name = get_kwarg_value(kwargs, 'row_name', '')  # Insert name for row
    rounding = get_kwarg_value(kwargs, 'rounding', 4)  # Specifies rounding (number of decimal places) to save array
    dim_total = size(array)  # Total numbers in array
    array = round(array, rounding)  # Rounding numbers in array
    array_reshaped_list = array.reshape(dim_total).tolist()  # Reshaping into 1D array and converting to list
    array_reshaped_list.insert(0, row_name)  # Insert row name to beginning of list
    with open(filename + '.csv', 'a') as csv_file:  # Opens .csv file
        writer_object = writer(csv_file, lineterminator='\n')  # Gets writing object from .csv file
        writer_object.writerow(array_reshaped_list)  # Writes observations as new row in .csv file
        csv_file.close()  # Closes .csv file

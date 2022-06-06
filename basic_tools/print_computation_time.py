"""
Computation time functions and print statements
"""


#######################################################
# Modules:
from time import time


#######################################################
# Print statement of the computation time:
def print_time_computation(initial_time, statement):
    final_time = time()  # Current time stamp
    time_difference = round(final_time - initial_time, 3)  # Time difference rounded to 3 decimal places
    print('Completed computing', statement, 'in', time_difference, 'seconds.')  # Print statement
    return final_time  # Returning current time stamp


#######################################################
# Decorator - prints the computation time of the decorated function with statement:
def timer(statement):
    def inner(function):
        def wrapper(*args, **kwargs):
            initial_time = time()  # Initial time stamp
            value = function(*args, **kwargs)  # Function evaluation
            print_time_computation(initial_time, statement)  # Print time with statement
            return value
        return wrapper
    return inner


#######################################################
# Function to print lines:
def print_lines():
    print('-------------------------------------------------------')


#######################################################
# Function to print lines:
def cls():
    print('\n'*30)

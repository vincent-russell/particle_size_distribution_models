"""
File saving and loading using pickle
"""


#######################################################
# Modules:
import pickle


#######################################################
# Function to save an object:
def save(object, filename):

    filehandler = open(filename + '.obj', 'wb')
    pickle.dump(object, filehandler)
    filehandler.close()


#######################################################
# Function to load an object:
def load(filename):

    filehandler = open(filename + '.obj', 'rb')
    output = pickle.load(filehandler)
    filehandler.close()

    return output

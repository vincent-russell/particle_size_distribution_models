"""

Title: Follows MNIST tensorflow tutorial for building neural network model
Author: Vincent Russell
Date: March 24, 2021

"""


#######################################################
# Modules:
import numpy as np
import tensorflow as tf
import time as tm
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


#######################################################
if __name__ == '__main__':

    #######################################################
    # Parameters:
    optimizer = 'Adam'  # Optimizer for training neural network
    epochs = 2  # Number of epochs (forward and backward passes) for training neural network


    #######################################################
    # Initialising timer for total computation:
    print(), print('Initialising...')
    initial_time = tm.time()  # Initial time stamp


    #######################################################
    # Import MNIST dataset:
    mnist = tf.keras.datasets.mnist


    #######################################################
    # Extract input and output data for training and testing:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()  # Extracts data into x,y pairs, 60000 for training and 10000 for testing
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalises pixel input data to space in (0, 1)


    #######################################################
    # Create neural network layers:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Softmax()  # Softmax layer to change logit ouputs to probablilies
    ])


    #######################################################
    # Creating loss function that takes output of model (logits) and returns a scalar loss:
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()


    #######################################################
    # Compiling model with optimizer and loss function:
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])


    #######################################################
    # Training neural network model on training set (60000 images):
    print(), print('Training model...')
    model.fit(x_train, y_train, epochs=epochs, verbose=1)


    #######################################################
    # Evaluating neural network model on test set (10000 images):
    print(), print('Evaluating model...')
    model.evaluate(x_test, y_test, verbose=1)


    #######################################################
    # Printing total computation time:
    computation_time = round(tm.time() - initial_time, 3)  # Initial time stamp
    print(), print('Total computation time:', str(computation_time), 'seconds.')  # Print statement


    #######################################################
    # Plotting some example images and predictions:

    # Extracting four random images from test set:
    random_numbers = np.random.randint(0, 10000, size=4)  # Random integers
    x_plot = np.zeros([4, 28, 28])  # Initialising random images
    y_plot = np.zeros(4)  # Initialising true values of random images
    y_pred = np.zeros([4])  # Initialising predicted values of random images
    for i in range(4):
        x_plot[i] = x_test[random_numbers[i]]  # Image
        y_plot[i] = y_test[random_numbers[i]]  # True value of image
        # Return most probable predicted value:
        y_pred[i] = np.argmax(model.predict(np.expand_dims(x_plot[i], 0)))

    # Creating figure and putting it on tkinter canvas:
    root = tk.Tk()
    fig = plt.Figure(figsize=(6.6, 5.6))
    fig.suptitle('Example of neural network predictions on four test images', fontsize=12)
    fig.subplots_adjust(hspace=0.4)
    root.geometry('660x560-3170+605')
    # Creating subplots:
    ax = fig.subplots(2, 2)
    ax[0, 0].imshow(x_plot[0])
    ax[0, 0].set_title('Predicted is ' + str(y_pred[0]) + ' and true is ' + str(y_plot[0]), fontsize=10)
    ax[0, 1].imshow(x_plot[1])
    ax[0, 1].set_title('Predicted is ' + str(y_pred[1]) + ' and true is ' + str(y_plot[1]), fontsize=10)
    ax[1, 0].imshow(x_plot[2])
    ax[1, 0].set_title('Predicted is ' + str(y_pred[2]) + ' and true is ' + str(y_plot[2]), fontsize=10)
    ax[1, 1].imshow(x_plot[3])
    ax[1, 1].set_title('Predicted is ' + str(y_pred[3]) + ' and true is ' + str(y_plot[3]), fontsize=10)
    # Canvas and toolbar:
    canvas = FigureCanvasTkAgg(fig, master=root)  # Adding canvas
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)  # Creating space for figure and toolbar on canvas
    NavigationToolbar2Tk(canvas, root)  # Adding navigation toolbar to figure
    # Final mainloop:
    tk.mainloop()

# Chapter 4 Test New ReLU class and Dense Layer class
# with training data from nnfs (spiral data)
# page 96-97
# Lori Pfahler
# January 2024

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# Dense Layer Class
class Layer_Dense:
    # Layer Initialization
    def __init__(self, n_inputs, n_neurons):
        # create data N(0, 1) data and multiply by 0.01 to keep numbers small
        # returns array with n_inputs rows and n_neurons columns
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # create an array filled with zeros with desired dimensions
        self.biases = np.zeros((1, n_neurons))
    # forward pass
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# ReLU Activation Function Class
class Activation_ReLU:
    # forward pass
    def forward(self, inputs):
        # calculate out values from inputs
        self.output = np.maximum(0, inputs)

# this sets the random seed among other things
nnfs.init()

# create dataset
X, y = spiral_data(samples = 100, classes = 3)

# create dense layer with 2 inputs features and 3 output value
dense1 = Layer_Dense(2, 3)

# create ReLU activation
activation1 = Activation_ReLU()

# make a forward pass with training data through this layer
dense1.forward(X)

# forward pass through activation function
# takes output of dense1.forward() and applies the ReLU activation function
activation1.forward(dense1.output)

# see output first 10 samples 
print(activation1.output[:10])

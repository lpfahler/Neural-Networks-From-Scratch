# Chapter 3 Dense Layer Class
# pages 66 - 71
# Lori Pfahler
# January 2024

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# this sets the random seed among other things
nnfs.init()

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



####### use new class

# create dataset, training data
X, y = spiral_data(samples = 100, classes = 3)

# create dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# perform a forward pass of our training data through this layer
dense1.forward(X)

print(dense1.output[:5])
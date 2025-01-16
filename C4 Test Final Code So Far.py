# Chapter 4 Test what we have so far!
# Layer_Dense, Activation_ReLU and Activation_Softmax Classes
# pages 108 - 109
# Lori Pfahler
# January 2024

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# include needed classes so far
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

# Softmax activation function
class Activation_Softmax:
    
    # define forward pass
    def forward(self, inputs):
        # get unnormalized probabilities 
        # subtract the largest input to avoid overflow in calculations of exp()
        # This will keep the results between zero and 1
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        # normalize for each sample
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        # return the probabilities
        self.output = probabilities


##### Test Classes/Code Developed so Far #####

##### Create Data and Needed Objects #####
# create dataset, training data
X, y = spiral_data(samples = 100, classes = 3)

# create dense layer #1 with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation 
activation1 = Activation_ReLU()

# create dense layer #2 with 3 input features and 3 output values
dense2 = Layer_Dense(3, 3)

# Create Softmax activation 
activation2 = Activation_Softmax()

##### Start Calculations #####
# perform a forward pass of our training data through this layer
dense1.forward(X)

# forward pass of activation function ReLU for dense layer #1
activation1.forward(dense1.output)

# forward pass through dense layer #2
dense2.forward(activation1.output)

# forward pass of activation function Softmax for dense layer #2
activation2.forward(dense2.output)

# results of output layer calculations - should be probabilities
# each case belongs in one of the three classes
print(activation2.output[:10])

# This output:
# [[0.33333334 0.33333334 0.33333334]
#  [0.33333316 0.3333332  0.33333364]
#  [0.33333287 0.3333329  0.33333418]
#  [0.3333326  0.33333263 0.33333477]
#  [0.33333233 0.3333324  0.33333528]
#  [0.33333284 0.33333287 0.3333343 ]
#  [0.33333182 0.3333319  0.33333626]
#  [0.33333182 0.3333319  0.33333623]
#  [0.3333315  0.3333316  0.33333692]
#  [0.33333105 0.3333312  0.33333772]]
#
# Shows that we have currently have random predictions
# for each observation - and all classes are equally likely
# since we are not current doing any parameter optimization
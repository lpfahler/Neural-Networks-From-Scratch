# Chapter 3 Adding Layers
# page 61
# Lori Pfahler
# January 2024

# this program will give the layer 2 outputs from
# a 4 input, 3 neuron layer 1, and 3 neuron layer 2
import numpy as np
import nnfs

# features (data) from the input layer
# four variables - three cases
inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

# weights and biases for first layer
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

# weights and biases for second layer
weights2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, -0.73, -0.13]]
biases2 = [-1.0, 2.0, -0.5]

# calculate layer 1 outputs
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases

# calculate layer 2 outputs
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer1_outputs)
print(layer2_outputs)


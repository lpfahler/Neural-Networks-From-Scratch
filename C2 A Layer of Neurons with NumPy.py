# Chapter 2 A Layer of Neurons with NumPy pp 42
# Lori Pfahler
# January 2024

import numpy as np

# four inputs to the neuron layer
inputs = inputs = [1, 2, 3, 2.5]
# weights for the three neurons in the layer in one list
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
# bias for the three neurons
biases = [2, 3, 0.5]

layer_outputs = np.dot(weights, inputs) + biases
print(layer_outputs)
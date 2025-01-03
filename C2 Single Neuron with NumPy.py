# Chapter 2 Single Neuron with NumPy p 40
# Lori Pfahler
# January 2025

import numpy as np

# example of one neuron with four inputs 
# our variables - most likely scaled/normed
inputs = [1, 2, 3, 2.5]
# weights for the four inputs - chosen randomly to start usually
weights = [0.2, 0.8, -0.5, 1]
# one bias value per neuron
bias = 2

outputs = np.dot(weights, inputs) + bias

print(outputs)
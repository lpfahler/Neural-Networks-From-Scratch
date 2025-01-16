# Chapter 4 Softmax Activation Function using NumPy
# pages 98 - 107
# Lori Pfahler
# January 2024

import numpy as np

# example of layer outputs from page 58 first row of output matrix
layer_outputs = [4.8, 1.21, 2.385]

# calculate the exponential value for layer_outputs
# using np.exp() function for arrays
exp_values = np.exp(layer_outputs)
print("exponentiated values")
print(exp_values)

# normalize values using array division
norm_values = exp_values / np.sum(exp_values)
print('Normalized esponentiated values:')
print(norm_values)

print('Sum of normalized values', np.sum(norm_values))
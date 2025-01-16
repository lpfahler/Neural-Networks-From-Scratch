# Chapter 4 Demo Matrix/Array Sums in Numpy
# pages 102
# Lori Pfahler
# January 2024

import numpy as np

# use layer_outputs to practice summing
layer_outputs = np.array([[4.8, 1.21, 2.385],
                          [8.9, -1.81, 0.2],
                          [1.41, 1.051, 0.026]])

print('sum using default axis')
print(np.sum(layer_outputs))
print('sum specifying axis = None - sums ALL numbers in the array')
print(np.sum(layer_outputs, axis = None))

print('using axis = 0 - sums COLUMNS of array')
print(np.sum(layer_outputs, axis = 0))

print('using axis = 1 - sums ROWS of array')
print(np.sum(layer_outputs, axis = 1))

print('using axis = 1 - sums ROWS of array and return a column vector')
print(np.sum(layer_outputs, axis = 1, keepdims = True))
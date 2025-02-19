# Chapter 2 A Layer of Neurons and Batch of Data with NumPy
# pages 54- 58
# Lori Pfahler
# January 2024

import numpy as np

inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

# just do matrix product
inputs_x_weights = np.dot(inputs, np.array(weights).T)
print(inputs_x_weights)

# calculate outputs
outputs = np.dot(inputs, np.array(weights).T) + biases
print(outputs)
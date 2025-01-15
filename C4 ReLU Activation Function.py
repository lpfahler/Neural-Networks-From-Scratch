# Chapter 4 ReLU Activation Function Code
# ReLU = Rectified Linear Activation Function
# pages 72 - 96
# Lori Pfahler
# January 2024
import numpy as np

inputs  = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

output = []
for i in inputs:
    if i > 0:
        output.append(i)
    else:
        output.append(0)
print(output)

# more simply written using max function
output = []
for i in inputs:
    output.append(max(0, i))
print(output)

# using maximum from numpy (an array/list based maximum)
output = np.maximum(0, inputs)
print(output)
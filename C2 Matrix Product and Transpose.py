# Chapter 2 Matrix Product and Transpose
# pages 44-53
# Lori Pfahler
# January 2024

import numpy as np
# (1, n) matrix a
a = np.array([[1, 2, 3]])
print(a)

# could also define a first as a list and then convert to matrix/array
a = [1, 2, 3]
a = np.array([a])
print(a)

# can also use expand_dims() method
a = [1, 2, 3]
a = np.expand_dims(np.array(a), axis = 0) # axis = 0 gives a row vector shape matrix
print(a)

# if we wanted a column vector matrix shape
a = [1, 2, 3]
a = np.expand_dims(np.array(a), axis = 1)
print(a)

# dot product of plain lists
a = [1, 2, 3]
b = [2, 3, 4]
# returns a value
print(np.dot(a, b))

# dot product of numpy arrays
a = np.array([a])
# need to transpose b
b = np.array([b]).T
# OR
# b = np.expand_dims(b, axis = 1)
# returns a (1, 1) array
print(np.dot(a, b))

# matrix product for larger matrices
# no specific matrix product function - use dot product

a = np.array([[0.49, 0.97, 0.53, 0.05],
              [0.33, 0.65, 0.62, 0.51],
              [1.00, 0.38, 0.61, 0.45],
              [0.74, 0.27, 0.64, 0.17],
              [0.36, 0.17, 0.96, 0.12]])
print(a)

b = np.array([[0.79, 0.32, 0.68, 0.90, 0.77],
              [0.18, 0.39, 0.12, 0.93, 0.09],
              [0.87, 0.42, 0.60, 0.71, 0.12],
              [0.45, 0.55, 0.40, 0.78, 0.81]])
print(b)

ab = np.dot(a, b)
print(ab)
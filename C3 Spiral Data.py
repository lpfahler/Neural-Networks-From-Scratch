# Chapter 3 Training Data
# page 62 - 65
# Lori Pfahler
# January 2024

# must use Python 3.12.2('Base':conda) - somehow I installed nnfs module in 
# the anaconda version?  Need to spend more time with VS Code and figuring out
# how to direct installation of modules to correct version of python
import nnfs.datasets
import numpy as np
import nnfs
# had to import this separately to use
from nnfs.datasets import spiral_data
# plotting module
import matplotlib.pyplot as plt

nnfs.init()

# create training data
X, y = spiral_data(samples = 100, classes = 3)

# plot of two x variables
plt.scatter(X[:,0], X[:, 1])
plt.show()

# the data
print(X)
print(y)

# plot of two x variables with classification - variable y (0, 1, or 2)
# shown in color
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = 'brg')
plt.show()
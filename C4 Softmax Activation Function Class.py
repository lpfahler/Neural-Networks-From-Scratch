# Chapter 4 Softmax Activation Function Class using NumPy 
# Allow Layer input from multiple neurons
# pages 98 - 107
# Lori Pfahler
# January 2024

import numpy as np

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
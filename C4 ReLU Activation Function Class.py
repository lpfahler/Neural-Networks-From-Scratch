# Chapter 4 ReLU Activation Function Code
# ReLU = Rectified Linear Activation Function Class
# pages 96-97
# Lori Pfahler
# January 2024

import numpy as np

# ReLU Activation Function Class
class Activation_ReLU:
    # forward pass
    def forward(self, inputs):
        # calculate out values from inputs
        self.output = np.maximum(0, inputs)

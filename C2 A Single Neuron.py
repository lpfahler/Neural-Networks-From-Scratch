# Chapter 2 A Single Neuron
# Lori Pfahler
# December 2024

# example of one neuron with three inputs 
# our variables - most likely scaled/normed
inputs = [1, 2, 3]
# weights for the three inputs - chosen randomly to start usually
weights = [0.2, 0.8, -0.5]
# one bias value per neuron
bias = 2
# output of the neuron
output = (inputs[0]*weights[0] + 
          inputs[1]*weights[1] +
          inputs[2]*weights[2] + 
          bias)

print(f'output = {output}')

# four inputs to the neuron
inputs = inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1]
output = (inputs[0]*weights[0] + 
          inputs[1]*weights[1] +
          inputs[2]*weights[2] + 
          inputs[3]*weights[3] +
          bias)

print(f'output = {output}')
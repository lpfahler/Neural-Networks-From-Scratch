# Chapter 2 A Layer of Neurons pp 30 - 34
# Using for loops with zip() function to calculate output
# Lori Pfahler
# December 2024

# four inputs to the neuron layer
inputs = inputs = [1, 2, 3, 2.5]
# weights for the three neurons in the layer in one list
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
# bias for the three neurons
biases = [2, 3, 0.5]

# output of current layer
layer_outputs = []

# loop over neurons
for neuron_weights, neuron_bias in zip(weights, biases):
    # added prints to see what was happening - I am new to zip()
    print(neuron_weights, neuron_bias)
    # set output of current neuron to zero
    neuron_output = 0

    # loop over inputs and this neuron's weights
    for n_input, weight in zip(inputs, neuron_weights):
        # multiply the inputs and weights
        # add to neuron's output variable
        print(n_input, weight)
        neuron_output += n_input*weight
    # add bias
    neuron_output += neuron_bias
    # put neuron's output in the layer's output list
    layer_outputs.append(neuron_output)
    print(layer_outputs)

print(layer_outputs)
# Chapter 4 Softmax Activation Function Simple Code
# pages 98 - 107
# Lori Pfahler
# January 2024

# example of layer outputs from page 58 first row of output matrix
layer_outputs = [4.8, 1.21, 2.385]

# define E Euler's Number - but could use math.e
# will use numpy exp function soon
E = 2.71828182846

# calculate the exponential value for layer_outputs
exp_values = []
for output in layer_outputs:
    exp_values.append(E ** output)
print("exponentiated values")
print(exp_values)

##### next update to code pg 100
# normalize values
norm_base = sum(exp_values)

norm_values = []
for value in exp_values:
    norm_values.append(value / norm_base)
print('Normalized esponentiated values:')
print(norm_values)

print('Sum of normalized values', sum(norm_values))

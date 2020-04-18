import numpy as np
import matplotlib.pyplot as plt
inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]


layer_outputs =[]

def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.05
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

X, y = create_data(100, 10)

plt.scatter(X[:,0], X[:,1], c=y)

plt.show()


for neuron_weights,  neuron_bias in zip(weights,biases):
    neuron_output = 0
    for input, weight in zip(inputs, neuron_weights):
        neuron_output += input * weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)



x = np.array(weights) @ np.array(inputs).T + np.array(biases).T


print (x)
print (layer_outputs)
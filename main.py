import random
import numpy as np


class Dense_layer():
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros((1, output_size))

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.biases
    
    def backward(self, gradient, output):
        self.weights += output.T.dot(gradient)
        self.biases += gradient


    def sigmoid(self, x):
        self.sigmoid_func = 1.0 / (1.0 + np.exp(-x))
        return self.sigmoid_func

    def d_sigmoid(self, x):
        self.d_sigmoid_func = np.multiply(x, 1.0-x)
        return self.d_sigmoid_func 


class Calculate_error:
    def calculate_output_error(self, real, expected):
        return np.subtract(expected, real)


input_to_hidden = Dense_layer(2, 4)
hidden_to_output = Dense_layer(4, 1)
Error_calc = Calculate_error()

data = [
    [
    np.array(([[1, 0]]), dtype=float),
    np.array(([[1]]), dtype=float)
    ],

    [
    np.array(([[1, 1]]), dtype=float),
    np.array(([[0]]), dtype=float)
    ],

    [
    np.array(([[0, 1]]), dtype=float),
    np.array(([[1]]), dtype=float)
    ],

    [
    np.array(([[0, 0]]), dtype=float),
    np.array(([[0]]), dtype=float)
    ]
]

learning_rate = 0.5


def feedForward(X):
    hidden_sigmoid = input_to_hidden.sigmoid(input_to_hidden.forward(X))
    output_sigmoid = input_to_hidden.sigmoid(hidden_to_output.forward(hidden_sigmoid))
    return output_sigmoid, hidden_sigmoid

def backprop(output_sigmoid, hidden_sigmoid):
    output_error = Y - output_sigmoid
    gradient = output_error * hidden_to_output.d_sigmoid(output_sigmoid)
    gradient = gradient * learning_rate

    hidden_error = gradient.dot(hidden_to_output.weights.T)
    hidden_gradient = hidden_error * hidden_to_output.d_sigmoid(hidden_sigmoid)
    hidden_gradient = hidden_gradient * learning_rate

    input_to_hidden.backward(hidden_gradient, X)
    hidden_to_output.backward(gradient, hidden_sigmoid)

def calculate_loss():
    loss = 0

    X, Y = data[0]
    output_sigmoid, hidden_sigmoid = feedForward(X)
    loss += Y - output_sigmoid

    X, Y = data[1]
    output_sigmoid, hidden_sigmoid = feedForward(X)
    loss += Y - output_sigmoid

    X, Y = data[2]
    output_sigmoid, hidden_sigmoid = feedForward(X)
    loss += Y - output_sigmoid

    X, Y = data[3]
    output_sigmoid, hidden_sigmoid = feedForward(X)
    loss += Y - output_sigmoid

    print(loss)


for i in range(20000):
    X, Y = random.choice(data)
    output_sigmoid, hidden_sigmoid = feedForward(X)
    backprop(output_sigmoid, hidden_sigmoid)

    if i % 10 == 0:
        calculate_loss()
    
    





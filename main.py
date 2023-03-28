import random
import matplotlib.pyplot as plt
import numpy as np
import pandas

class Dense_layer():
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros((1, output_size))

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.biases
    
    def backward(self, gradient, learning_rate):
        gradient_delta = gradient * self.d_sigmoid(self.sigmoid_func)
        gradient_delta = gradient_delta * learning_rate

        self.weights += self.input.T.dot(gradient_delta)
        self.biases += gradient_delta

        return gradient_delta.dot(self.weights.T)

    def sigmoid(self, x):
        self.sigmoid_func = 1.0 / (1.0 + np.exp(-x))
        return self.sigmoid_func

    def d_sigmoid(self, x):
        self.d_sigmoid_func = np.multiply(x, 1.0-x)
        return self.d_sigmoid_func


class Network:
    def __init__(self):
        self.learning_rate = 0.5
        self.setup = [
            Dense_layer(784, 64),
            Dense_layer(64, 10)
        ]
        self.data = [
                    [np.array(([[1, 0]]), dtype=float),
                     np.array(([[1]]), dtype=float)
                    ],
                    [np.array(([[1, 1]]), dtype=float),
                     np.array(([[0]]), dtype=float)
                    ],
                    [np.array(([[0, 1]]), dtype=float),
                     np.array(([[1]]), dtype=float)
                    ],
                    [np.array(([[0, 0]]), dtype=float),
                     np.array(([[0]]), dtype=float)
                    ]
                ]


    def train(self, X, Y):
        output = X
        for layer in self.setup:
            output = layer.sigmoid(layer.forward(output))

        gradient = Y - output
        for layer in reversed(self.setup):
            gradient = layer.backward(gradient, self.learning_rate)

        return output


    def predict(self, index):
        return self.train(self.data[index][0] , self.data[index][1])


    def train_loop(self, num):
        for i in range(0, num):
            X, Y = random.choice(self.data)
            self.train(X, Y)

    def one_hot_encoding(self, Y):
        pass




nn = Network()
nn.train_loop(10000)

print(nn.predict(0), "   1")
print(nn.predict(1), "   0")
print(nn.predict(2), "   1")
print(nn.predict(3), "   0")
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
np.set_printoptions(suppress=True)

class Dense_layer():
    def __init__(self, input_size, output_size, acctivation="sigmoid"):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros((1, output_size))
        self.activation = acctivation

    def forward(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.biases

        if self.activation == "sigmoid":
            return self.sigmoid(self.output)
        elif self.activation == "relu":
            return self.relu(self.output)
        elif self.activation == "softmax":
            return self.softmax(self.output)
    
    def backward(self, gradient, learning_rate):
        if self.activation == "sigmoid":
            gradient_delta = gradient * self.d_sigmoid(self.sigmoid_func)

        elif self.activation == "relu":
            gradient_delta = 1 / m * gradient.dot(self.output.T)

        elif self.activation == "softmax":
            gradient_delta = gradient * self.d_softmax(self.softmax_func)

        self.weights += learning_rate * self.input.T.dot(gradient_delta)
        self.biases += learning_rate * gradient_delta

        if self.activation == "relu":
            return gradient.dot(self.weights.T)
        return gradient_delta.dot(self.weights.T)


    def sigmoid(self, x):
        self.sigmoid_func = 1.0 / (1.0 + np.exp(-x))
        return self.sigmoid_func

    def d_sigmoid(self, x):
        self.d_sigmoid_func = np.multiply(x, 1.0-x)
        return self.d_sigmoid_func
    
    def relu(self, x):
        self.relu_func = np.maximum(x, 0)
        return self.relu_func

    def d_relu(self, x):
        return x > 0
    
    def softmax(self, x):
        self.exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.softmax_func = self.exps / np.sum(self.exps, axis=1, keepdims=True)
        return self.softmax_func
    
    def d_softmax(self, x):
        return x * (1 - x)


class Network:
    def __init__(self):
        self.learning_rate = 0.05
        self.setup = [
            Dense_layer(784, 16),
            Dense_layer(16, 10, "softmax")
        ]

    def train(self, X, Y):
        output = X
        for layer in self.setup:
            output = layer.forward(output)

        gradient = self.one_hot_encoding(Y) - output

        for layer in reversed(self.setup):
            gradient = layer.backward(gradient, self.learning_rate)

        return output

    def predict(self, X, Y):
        output = X
        for layer in self.setup:
            output = layer.forward(output)

        return np.argmax(output)
        
    def train_loop(self, num):
        for i in range(0, num):
            X, Y = random.choice(self.data)
            self.train(X, Y)

    def one_hot_encoding(self, Y):
        one_hot = np.zeros(10)
        one_hot[Y] = 1
        return one_hot

def checkAccuaracy():
    wrong = 0
    for Y, X in dataSetTest:
        final = nn.predict(X, Y)

        if Y != final:
            wrong += 1
    
    print(wrong)
    print(100 - wrong / len(dataSetTest) * 100)


nn = Network()
data = pd.read_csv('project/train.csv')
data = np.array(data)
np.random.shuffle(data)

m, n = data.shape
data_train = data[:40000]
data_test = data[40000:]

Y_train = data_train[:, 0]
X_train = data_train[:, 1:]
X_train = X_train / 255

Y_test = data_test[:, 0]
X_test = data_test[:, 1:]
X_test = X_test / 255

dataSetTrain = []
dataSetTest = []

for i,label in enumerate(Y_train):
    dataSetTrain.append([label, np.array([list(X_train[i])])])

for i,label in enumerate(Y_test):
    dataSetTest.append([label, np.array([list(X_test[i])])])

epochs = 10
for i in range(epochs):
    np.random.shuffle(dataSetTrain)

    for Y, X in dataSetTrain:
        nn.train(X, Y)

    print(i, " Epoch")
    checkAccuaracy()
    # nn.learning_rate -= nn.learning_rate * 0.1



# no epochs = 10 epochs
# best - 86,6% acc (15 hidden / 0.1 lr)
# best - 86,95% acc (32 hidden / 0.1 lr)
# best - 87,15% acc (16 hidden / 0.1 lr)
# best - 89,55% acc (16 hidden but 25 epochs / 0.1 lr)
# best - 91,3% acc (16 hidden / 0.2 lr)
# best - 92% acc (16 hidden but 25 epochs / 0.2 lr)
# best - 92.7% acc (32 hidden / 16 hidden / 0.5 lr - 10%)
# best - 93.5% acc (16 hidden 0.5 lr - 10%)

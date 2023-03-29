import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
np.set_printoptions(suppress=True)

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
        self.learning_rate = 0.1
        self.setup = [
            Dense_layer(784, 16),
            Dense_layer(16, 10),
        ]


    def train(self, X, Y):
        output = X
        for layer in self.setup:
            output = layer.sigmoid(layer.forward(output))

        gradient = self.one_hot_encoding(Y) - output

        for layer in reversed(self.setup):
            gradient = layer.backward(gradient, self.learning_rate)

        return output


    def predict(self, X, Y):
        output = X
        for layer in self.setup:
            output = layer.sigmoid(layer.forward(output))

        return np.argmax(output)
        


    def train_loop(self, num):
        for i in range(0, num):
            X, Y = random.choice(self.data)
            self.train(X, Y)

    def one_hot_encoding(self, Y):
        one_hot = np.zeros(10)
        one_hot[Y] = 1
        return one_hot


nn = Network()


data = pd.read_csv('project/train.csv')
data = np.array(data)
np.random.shuffle(data)

m, n = data.shape
data_train = data[:40000]
data_test = data[40000:]

print(data_test.shape)

Y_train = data_train[:, 0]
X_train = data_train[:, 1:] # x je 1D np array skusit to dát do 2D array pls work
X_train = X_train / 255

Y_test = data_test[:, 0]
X_test = data_test[:, 1:] # x je 1D np array skusit to dát do 2D array pls work
X_test = X_test / 255

dataSetTrain = []
dataSetTest = []

for i,label in enumerate(Y_train):
    dataSetTrain.append([label, np.array([list(X_train[i])])])

for i,label in enumerate(Y_test):
    dataSetTest.append([label, np.array([list(X_test[i])])])

epochs = 10
for _ in range(epochs):
    np.random.shuffle(dataSetTrain)

    for Y, X in dataSetTrain:
        nn.train(X, Y)



wrong = 0
for Y, X in dataSetTest:
    final = nn.predict(X, Y)

    if Y != final:
        wrong += 1

    # image = X.reshape((28,28)) * 255
    # plt.gray()
    # plt.imshow(image, interpolation="nearest")
    # plt.show()

print(wrong)




# best - 86,6% acc (15 hidden)
# best - 86,95% acc (32 hidden)
# best - 87,15% acc (16 hidden)
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
    
    def backward(self, gradient, learning_rate):
        if self.activation == "sigmoid":
            gradient_delta = gradient * self.d_sigmoid(self.sigmoid_func)
        elif self.activation == "relu":
            gradient_delta = gradient * self.d_relu(self.relu_output)
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
    
    def relu(self, x):
        self.relu_output = np.maximum(0, x) 
        return self.relu_output

    def d_relu(self, x):
        return 1 * (x > 0)


class Network:
    def __init__(self):
        self.learning_rate = 0.5
        self.setup = [
            Dense_layer(784, 16),
            Dense_layer(16, 10),
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

epochs = 25
for _ in range(epochs):
    np.random.shuffle(dataSetTrain)

    for Y, X in dataSetTrain:
        nn.train(X, Y)

    nn.learning_rate -= nn.learning_rate * 0.2



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



# no epochs = 10 epochs
# best - 86,6% acc (15 hidden / 0.1 lr)
# best - 86,95% acc (32 hidden / 0.1 lr)
# best - 87,15% acc (16 hidden / 0.1 lr)
# best - 89,55% acc (16 hidden but 25 epochs / 0.1 lr)
# best - 91,3% acc (16 hidden / 0.2 lr)
# best - 92% acc (16 hidden but 25 epochs / 0.2 lr)
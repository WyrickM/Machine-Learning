########################################################################
# Mantz Wyrick
# Machine Learning Homework #5
#
# Neural networks. Specifically back propogation
#
########################################################################


from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from random import random
import numpy as np


# function to intialize the network
# just setting up a structure for the network that is a list of layers
# each layer (hidden, output) just a bunch of weights that will go into my
# network structure each layer is appended to that network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()  # intialize weights to random number in [0..1]
    hidden_layer = [
        {"weights": [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)
    ]
    network.append(hidden_layer)
    output_layer = [
        {"weights": [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)
    ]
    network.append(output_layer)
    return network


def forward_propagate(network, X):
    inputs = X
    for layer in network:
        new_inputs = []
        for node in layer:
            activation = activate(node["weights"], X)  # need activate function
            node["output"] = transfer(activation)  # need transfer function
            new_inputs.append(node["output"])  # output of node input to another
        inputs = new_inputs
    return inputs  # return output from last layer


def activate(weights, inputs):
    activation = weights[-1]  # bias weight, last weight in the vector
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


def transfer(activation):
    return np.tanh(activation)


# going back through the layers one layer at a time
# going to determine the gradient and then update the weights
def back_propagate_error(network, expected):
    for i in reversed(range(len(network))):  # from output back to input layers
        layer = network[i]
        errors = list()
        if i != len(network) - 1:  # not the output layer
            for j in range(len(layer)):
                error = 0.0
                for node in network[i + 1]:
                    error += node["weights"][j] * node["delta"]
                errors.append(error)
        else:  # output layer
            for j in range(len(layer)):
                node = layer[j]
                errors.append(expected[j] - node["output"])
        for j in range(len(layer)):
            node = layer[j]
            node["delta"] = errors[j] * transfer_derivative(node["output"])


def transfer_derivative(output):
    return 1.0 - np.tanh(output) ** 2


def update_weights(network, row, eta):
    for i in range(len(network)):
        inputs = row[:]
        if i != 0:
            inputs = [node["output"] for node in network[i - 1]]
        for node in network[i]:
            for j in range(len(inputs)):
                node["weights"][j] += eta * node["delta"] * inputs[j]
            node["weights"][-1] += eta * node["delta"]


def train_network(network, X, y, eta, num_epochs, num_outputs):
    expected = np.full((2), 0)
    for epoch in range(num_epochs):
        sum_error = 0
        for i in range(len(y)):
            outputs = forward_propagate(network, X[i])
            # expected = [0 for i in range(num_outputs)]
            # expected[y[i]] = 1
            if y[i] == -1:
                expected[0] = 1
                expected[1] = 0
            else:
                expected[0] = 0
                expected[1] = 1
            sum_error += sum(
                [(expected[i] - outputs[i]) ** 2 for i in range(len(expected))]
            )
            back_propagate_error(network, expected)
            update_weights(network, X[i], eta)
        print(">epoch=%d, lrate=%.3f, error=%.3f" % (epoch, eta, sum_error))


# have to forward propogate the feature values
# into an output, just use it to give a label
def test_network(network, X_test, y_test, n_outputs):
    expected = np.full((2), 0)
    sum_error = 0
    for i in range(len(y_test)):
        outputs = forward_propagate(network, X_test[i])
        # expected = [0 for i in range(n_outputs)]
        # expected[y_test[i]] = 1
        if y[i] == -1:
            expected[0] = 1
            expected[1] = 0
        else:
            expected[0] = 0
            expected[1] = 1
        sum_error += sum(
            [(expected[i] - outputs[i]) ** 2 for i in range(len(expected))]
        )
    print("Results from test_network")
    print("mean squared error = ", sum_error / float(len(y_test)))


if __name__ == "__main__":
    features, targets = fetch_openml("mnist_784", version=1, return_X_y=True)
    X = []
    y = []
    for i in range(len(targets)):
        if targets[i] == "1" or targets[i] == "0":
            X.append(features[i])
            if targets[i] == "0":
                y.append(-1)
            else:
                y.append(1)
    n_inputs = len(X[0])
    n_outputs = 2
    # print('y:', y)
    # print('X:', X)

    network = initialize_network(n_inputs, 2, n_outputs)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.67, test_size=0.33
    )
    for layer in network:
        print("layer \n", layer)
    train_network(network, X_train, y_train, 0.1, 10, n_outputs)
    for layer in network:
        print("layer \n", layer)

    print()
    test_network(network, X_test, y_test, n_outputs)

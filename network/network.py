import numpy as np
from random import shuffle


class Network:
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        print('lalala')

    def SGD(self, training_data, epochs, batch_size, eta, test_data=None):
        n = len(training_data)
        for _ in range(epochs):
            shuffle(training_data)
            mini_batches = [training_data[i:i+batch_size] for i in range(0, n, batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

    def update_mini_batch(self, mini_batch, eta):
        test_data = np.random.random((1, 2))
        outputs = np.random.random((1, 2))

        self.backprop(test_data, outputs)


    @staticmethod
    def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def backprop(self, x, y):
        """
        x - training inputs
        y - outputs from training data
        """
        activation = x
        activations = [x]
        zs_vectors = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs_vectors.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        for l in range(2, self.num_layers):
            z = zs_vectors[-l]
            sp = self.sigmoid_derivative(z)










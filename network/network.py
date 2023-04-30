import numpy as np
from random import shuffle


class Network:
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

    def SGD(self, training_data, epochs, batch_size, eta, test_data=None):
        n = len(training_data)
        for _ in range(epochs):
            shuffle(training_data)
            mini_batches = [training_data[i:i+batch_size] for i in range(0, n, batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

    def update_mini_batch(self, mini_batch, eta):
        pass





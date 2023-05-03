import numpy as np
from random import shuffle


class Network:
    def __init__(self, layer_sizes):
        """
        weights represented as array of matrices, where in these matrices w_jk element it's a connection between
        k-th neuron in 'l' layer and j-th neuron in 'l+1' layer
        """
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, batch_size, eta, test_data=None):
        n = len(training_data)
        n_test = 0
        if test_data:
            n_test = len(test_data)

        for j in range(epochs):
            shuffle(training_data)
            mini_batches = [training_data[i:i + batch_size] for i in range(0, n, batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")

    def evaluate(self, data):
        results = [(np.argmax(self.feedforward(x)), y) for x, y in data]
        return sum(int(x == np.where(y == 1.0)[0]) for x,y in results)

    def update_mini_batch(self, mini_batch, eta):
        nabla_weights, nabla_biases = [np.zeros(w.shape) for w in self.weights], [np.zeros(b.shape) for b in self.biases]
        for x, y in mini_batch:
            delta_nabla_weights, delta_nabla_biases = self.backprop(x, y)
            nabla_weights = [nw+dnw for nw, dnw in zip(nabla_weights, delta_nabla_weights)]
            nabla_biases = [nb+dnb for nb, dnb in zip(nabla_biases, delta_nabla_biases)]

        self.weights = [w - (eta/len(mini_batch)) * derivative_w for w, derivative_w in zip(self.weights, nabla_weights)]
        self.biases = [b - (eta/len(mini_batch)) * derivative_b for b, derivative_b in zip(self.biases, nabla_biases)]

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    @staticmethod
    def cost_function_derivative(output_activations, y):
        """
        derivative of 0.5*(y - output_activations)^2 = output_activations - y
        """
        return output_activations - y

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

        nabla_weights, nabla_biases = [np.zeros(w.shape) for w in self.weights], [np.zeros(b.shape) for b in self.biases]

        delta = self.cost_function_derivative(activations[-1], y) * self.sigmoid_derivative(zs_vectors[-1])

        nabla_biases[-1] = delta
        nabla_weights[-1] = delta * activations[-2].transpose()

        # iterate backwards(layer = layer from end)
        for layer in range(2, self.num_layers):
            z = zs_vectors[-layer]
            delta = np.dot(self.weights[-layer+1].transpose(), delta) * self.sigmoid_derivative(z)
            nabla_weights[-layer] = np.dot(delta, activations[-layer-1].transpose())
            nabla_biases[-layer] = delta
        return nabla_weights, nabla_biases


from network.network import *
from mnist.loader import *

training_data = list(get_data())
test_data = list(get_test_data())

net = Network([784, 30, 10])
net.SGD(training_data, 10, 3, 0.1, test_data)





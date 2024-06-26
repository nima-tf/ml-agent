import numpy as np
import matplotlib.pyplot as plt


class Layer:
    def __lnit__(self) -> None:
        self.input = None
        self.output = None
    
    def forward_propagation(self, input):
        raise NotImplementedError
    
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError

class FCLayer(Layer):
    def __init__(self, input_size, outpus_size) -> None:
        self.weights = np.random.rand(input_size, outpus_size) - 0.5
        self.bias = np.random.rand(1, outpus_size) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
    
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime) -> None:
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output
    
    def backward_propagation(self, output_error, learning_rate) -> None:
        return self.activation_prime(self.input) * output_error
    




if __name__ == "__main__":
    pass


class Network:
    def __init__(self) -> None:
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)
    
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        sample_size = len(input_data)
        result = []
        for i in range(sample_size):
            for layer in self.layers:
                output = layer.forward_propagation(input_data[i])
            result.append(output)
        return result
    
    def fit(self, x_train, y_train, epochs, learning_rate):
        sample_size = len(x_train)
        for i in range(epochs):
            err = 0
            for j in range(sample_size):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                err += self.loss(y_train[j], output)
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
            err /= sample_size
            print('epoch {0}/{1}   error={2}'.format(i+1, epochs, err))

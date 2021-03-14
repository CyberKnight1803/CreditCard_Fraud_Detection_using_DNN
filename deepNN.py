import numpy as np
from layers import Layer
from loss_functions import CrossEntropy

class DNN():

    def __init__(self, layer_dims, lRate=1, n_iters=5000, activation='ReLu'):
        self.lRate = lRate
        self.n_iters = n_iters
        self.loss = CrossEntropy()

        self.layers = []
        self.n_layers = len(layer_dims) - 1

        #Initializing all layers with ReLu except last
        for l in range(1, self.n_layers):
            layer_dim = (layer_dims[l], layer_dims[l - 1])
            self.layers.append(Layer(l, layer_dim, activation))
            print(self.layers[l - 1].__str__())
        
        layer_dim = (layer_dims[self.n_layers], layer_dims[self.n_layers - 1]) 
        self.layers.append(Layer(self.n_layers, layer_dim, 'Sigmoid'))
        print(self.layers[self.n_layers - 1].__str__())

    def forward_propagation(self, X):
        A = X
        caches = []

        for layer in self.layers:
            _A = A
            A, Z = layer.forward_pass(_A)

            caches.append((_A, Z))

        return A, caches

    def compute_cost(self, y, AL):
        m = y.shape[1]
        J = - np.sum(self.loss(y, AL)) / m

        return J

    def backward_propagation(self, AL, y, caches):
        dAL = self.loss.derivative(y, AL)

        _dA = dAL
        for l in reversed(range(self.n_layers)):
            dA = _dA
            _dA = self.layers[l].backward_pass(dA, caches[l])
    
    def fit(self, X, y, print_cost=False):
        costs = []

        for i in range(self.n_iters):
            AL, caches = self.forward_propagation(X)
            cost = self.compute_cost(y, AL)
            self.backward_propagation(AL, y, caches)

            for layer in self.layers:
                layer.update_params(self.lRate)

            if print_cost and i % 200 == 1:
                print(f"Cost after iteration{i + 1}: {cost}")
                costs.append(cost)
    
    def accuracy(self, X, y):
        A, caches = self.forward_propagation(X)
        
        P = np.around(A)
        acc = np.sum(P == y) / y.shape[1]
        return acc


import numpy as np
from layers import Layer
from loss_functions import CrossEntropy
from optimizers import optimizers


class DNN():

    def __init__(self, layer_dims, momentum=0, lRate=0.45, n_iters=5000, activation='ReLu', initializer = 'He', optimizer='StochasticGD'):
        self.lRate = lRate
        self.n_iters = n_iters
        self.loss = CrossEntropy()
        self.optimizer = optimizers[optimizer](self.lRate, momentum)

        self.layers = []
        self.n_layers = len(layer_dims) - 1

        #Initializing all layers with ReLu except last
        for l in range(1, self.n_layers):
            layer_shape = (layer_dims[l], layer_dims[l - 1])
            self.layers.append(Layer(l, layer_shape, activation, initializer))
            print(self.layers[l - 1].__str__())
        
        layer_shape = (layer_dims[self.n_layers], layer_dims[self.n_layers - 1]) 
        self.layers.append(Layer(self.n_layers, layer_shape, 'Sigmoid', 'Random'))
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

        J = - np.sum(self.loss(y, AL)) / self.m
        return J

    def backward_propagation(self, AL, y, caches):
        dAL = self.loss.derivative(y, AL)

        _dA = dAL
        for l in reversed(range(self.n_layers)):
            dA = _dA
            _dA = self.layers[l].backward_pass(dA, caches[l])

    def fit(self, X, y, print_cost=False):
        self.m = X.shape[1]
        costs = []
        mechanism = {
            'forward_prop' : self.forward_propagation,
            'backward_prop' : self.backward_propagation,
            'compute_cost' : self.compute_cost
        }

        for i in range(0, self.n_iters):
            self.optimizer(X, y, self.layers, mechanism, costs, i, print_cost=True)

        return costs
    
    def accuracy(self, X, y):
        A, caches = self.forward_propagation(X)
        
        P = np.around(A)
        acc = np.sum(P == y) / y.shape[1]
        return acc


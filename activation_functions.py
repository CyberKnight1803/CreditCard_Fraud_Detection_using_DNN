# All Activation functions required for deep learning
import numpy as np

class BinaryStep():
    def __call__(self, z):
        return np.heaviside(z, 1)


class Sigmoid():
    def __call__(self, z):
        S = 1 / (1 + np.exp(-z))
        return S
    
    def derivative(self, z):
        dz = self.__call__(z) * (1 - self.__call__(z))
        return dz

class TanH():
    def __call__(self, z):
        T = np.tanh(z)
        return T
    
    def derivative(self, z):
        dz = 1 - np.power(self.__call__(z), 2)
        return dz

class ReLu():
    def __call__(self, z):
        R = np.maximum(0, z)
        return R
    
    def derivative(self, z):
        dz = np.where(z > 0, 1, 0)
        return dz

class LeakyReLu():
    def __init__(self, grad = 0.15):
        self.grad = grad
    
    def __call__(self, z):
        L = np.where(z > 0, z, self.grad * z)
        return L
    
    def derivative(self, z):
        dz = np.where(z > 0, 1, self.grad)
        return dz


activations = {
    'BinarStep': BinaryStep,
    'Sigmoid': Sigmoid,
    'TanH': TanH,
    'ReLu': ReLu,
    'LeakyReLu': LeakyReLu,
    }
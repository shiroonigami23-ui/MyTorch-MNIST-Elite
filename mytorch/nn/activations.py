
import numpy as np
from mytorch.nn.module import Module

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, Z):
        # Store Z for the backward pass
        self.cache = Z
        # Return max(0, Z)
        return np.maximum(0, Z)

    def backward(self, dL_dA):
        # dL_dA is the gradient coming from the next layer
        Z = self.cache

        # The gradient of ReLU is 1 for Z > 0 and 0 otherwise
        dZ = np.array(dL_dA, copy=True)
        dZ[Z <= 0] = 0

        return dZ

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, Z):
        # Sigmoid formula: 1 / (1 + exp(-Z))
        A = 1 / (1 + np.exp(-Z))
        self.cache = A # For Sigmoid, caching the output is more efficient
        return A

    def backward(self, dL_dA):
        A = self.cache
        # Derivative of Sigmoid: A * (1 - A)
        dZ = dL_dA * (A * (1 - A))
        return dZ


import numpy as np

class SGD:
    def __init__(self, layers, lr=0.1, momentum=0.9):
        # 'layers' should be a list of all layers that have parameters (Linear layers)
        self.layers = layers
        self.lr = lr
        self.mu = momentum
        # Dictionary to store the velocity (moving average of gradients)
        self.vW = {id(layer): np.zeros_like(layer.W) for layer in layers if hasattr(layer, 'W')}
        self.vb = {id(layer): np.zeros_like(layer.b) for layer in layers if hasattr(layer, 'b')}

    def step(self):
        # Iterate through layers and update W and b using the gradients stored in them
        for layer in self.layers:
            if hasattr(layer, 'W') and hasattr(layer, 'dW'):
                lid = id(layer)
                # v = mu * v - lr * grad
                self.vW[lid] = self.mu * self.vW[lid] - self.lr * layer.dW
                self.vb[lid] = self.mu * self.vb[lid] - self.lr * layer.db

                # Update weights
                layer.W += self.vW[lid]
                layer.b += self.vb[lid]

    def zero_grad(self):
        # In PyTorch, gradients accumulate by default.
        # We need to reset them to zero before the next backward pass.
        for layer in self.layers:
            if hasattr(layer, 'dW'):
                layer.dW.fill(0.0)
                layer.db.fill(0.0)

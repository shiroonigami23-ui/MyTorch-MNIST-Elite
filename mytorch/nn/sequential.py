
from mytorch.nn.module import Module

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        # Forward pass: Pass data through each layer in order
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, dL_dy):
        # Backward pass: Pass gradient through each layer in REVERSE order
        # dL_dy is the gradient from the loss function
        for layer in reversed(self.layers):
            dL_dy = layer.backward(dL_dy)
        return dL_dy

    def __getitem__(self, i):
        # Allows accessing layers like model[0]
        return self.layers[i]

    def __len__(self):
        return len(self.layers)

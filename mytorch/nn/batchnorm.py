
import numpy as np
from mytorch.nn.module import Module

class BatchNorm(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super().__init__()
        self.eps = eps
        self.momentum = momentum

        # Trainable parameters
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))

        # Gradients
        self.dgamma = None
        self.dbeta = None

        # Running averages for inference (to be used later)
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))

    def forward(self, x):
        # x shape: (Batch_Size, Num_Features)
        batch_mean = np.mean(x, axis=0, keepdims=True)
        batch_var = np.var(x, axis=0, keepdims=True)

        # Normalize
        x_hat = (x - batch_mean) / np.sqrt(batch_var + self.eps)

        # Scale and Shift
        y = self.gamma * x_hat + self.beta

        # Cache for backward pass
        self.cache = (x, x_hat, batch_mean, batch_var)

        return y

    def backward(self, dL_dy):
        x, x_hat, batch_mean, batch_var = self.cache
        N = x.shape[0]

        # Gradients w.r.t. parameters
        self.dgamma = np.sum(dL_dy * x_hat, axis=0, keepdims=True)
        self.dbeta = np.sum(dL_dy, axis=0, keepdims=True)

        # Gradient w.r.t. input (dL/dx) - The complex part
        dx_hat = dL_dy * self.gamma
        dvar = np.sum(dx_hat * (x - batch_mean) * -0.5 * (batch_var + self.eps)**-1.5, axis=0, keepdims=True)
        dmean = np.sum(dx_hat * -1 / np.sqrt(batch_var + self.eps), axis=0, keepdims=True) +                 dvar * np.sum(-2 * (x - batch_mean), axis=0, keepdims=True) / N

        dL_dx = dx_hat / np.sqrt(batch_var + self.eps) +                 dvar * 2 * (x - batch_mean) / N +                 dmean / N

        return dL_dx

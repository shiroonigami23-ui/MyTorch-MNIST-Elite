
import numpy as np
from mytorch.nn.module import Module

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        # Kaiming/He Initialization (Better for staying ahead in accuracy)
        # This keeps the variance of activations consistent across layers.
        limit = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(out_features, in_features) * limit
        self.b = np.zeros((out_features, 1))

        # Real storage for gradients - these will be populated during .backward()
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, A):
        # A: (Batch_Size, In_Features)
        self.cache = A

        # Z = A @ W.T + b.T
        # Output: (Batch_Size, Out_Features)
        return np.dot(A, self.W.T) + self.b.T

    def backward(self, dL_dZ):
        # dL_dZ: The gradient of the loss with respect to the output of this layer
        # Shape: (Batch_Size, Out_Features)

        A = self.cache # Retrieve input from forward pass
        batch_size = A.shape[0]

        # 1. Gradient w.r.t Weights: dL/dW = dL/dZ^T @ A
        # Result shape must match self.W: (Out_Features, In_Features)
        self.dW = np.dot(dL_dZ.T, A) / batch_size

        # 2. Gradient w.r.t Bias: dL/db = sum of dL/dZ across the batch
        # Result shape must match self.b: (Out_Features, 1)
        self.db = np.sum(dL_dZ.T, axis=1, keepdims=True) / batch_size

        # 3. Gradient w.r.t Input (to pass to the previous layer): dL/dA = dL/dZ @ W
        # Result shape must match A: (Batch_Size, In_Features)
        dL_dA = np.dot(dL_dZ, self.W)

        return dL_dA

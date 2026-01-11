
import numpy as np

class CrossEntropyLoss:
    def __init__(self, smoothing=0.1):
        self.smoothing = smoothing
        self.cache = None

    def forward(self, Z, y):
        batch_size, num_classes = Z.shape
        # Numerical stability
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        probs = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

        # Label Smoothing: Create soft labels
        # Instead of [0, 1, 0], we get [0.03, 0.93, 0.03]
        y_true = np.zeros_like(probs)
        y_true.fill(self.smoothing / (num_classes - 1))
        y_true[np.arange(batch_size), y.flatten()] = 1.0 - self.smoothing

        self.cache = (probs, y_true)

        # Cross-Entropy with soft labels
        loss = -np.sum(y_true * np.log(probs + 1e-12)) / batch_size
        return loss

    def backward(self):
        probs, y_true = self.cache
        batch_size = probs.shape[0]
        # The gradient for smoothed CE is simply (Probs - Soft_Labels)
        return (probs - y_true) / batch_size

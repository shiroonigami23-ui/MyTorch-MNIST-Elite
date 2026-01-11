
import numpy as np

class Adam:
    def __init__(self, layers, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0001):
        self.layers = layers
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m_W = {id(layer): np.zeros_like(layer.W) for layer in layers if hasattr(layer, 'W')}
        self.v_W = {id(layer): np.zeros_like(layer.W) for layer in layers if hasattr(layer, 'W')}
        self.m_b = {id(layer): np.zeros_like(layer.b) for layer in layers if hasattr(layer, 'b')}
        self.v_b = {id(layer): np.zeros_like(layer.b) for layer in layers if hasattr(layer, 'b')}

    def step(self):
        self.t += 1
        for layer in self.layers:
            if hasattr(layer, 'W') and hasattr(layer, 'dW'):
                lid = id(layer)

                # Apply Weight Decay (AdamW style)
                grad_W = layer.dW + self.weight_decay * layer.W
                grad_b = layer.db

                self.m_W[lid] = self.beta1 * self.m_W[lid] + (1 - self.beta1) * grad_W
                self.m_b[lid] = self.beta1 * self.m_b[lid] + (1 - self.beta1) * grad_b

                self.v_W[lid] = self.beta2 * self.v_W[lid] + (1 - self.beta2) * (grad_W**2)
                self.v_b[lid] = self.beta2 * self.v_b[lid] + (1 - self.beta2) * (grad_b**2)

                m_W_hat = self.m_W[lid] / (1 - self.beta1**self.t)
                m_b_hat = self.m_b[lid] / (1 - self.beta1**self.t)
                v_W_hat = self.v_W[lid] / (1 - self.beta2**self.t)
                v_b_hat = self.v_b[lid] / (1 - self.beta2**self.t)

                layer.W -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.eps)
                layer.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)

    def zero_grad(self):
        for layer in self.layers:
            if hasattr(layer, 'dW'):
                layer.dW.fill(0.0)
                layer.db.fill(0.0)

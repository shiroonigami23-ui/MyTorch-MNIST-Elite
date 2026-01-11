
import numpy as np
from mytorch.nn.module import Module

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p # Probability of DROPPING a neuron
        self.mask = None

    def forward(self, x):
        # We create a mask of 1s and 0s
        # We scale the output by 1/(1-p) so that the expected value remains the same
        # during training and inference (Inverted Dropout)
        keep_prob = 1 - self.p
        self.mask = (np.random.rand(*x.shape) < keep_prob) / keep_prob

        return x * self.mask

    def backward(self, dL_dy):
        # We only pass gradients back through the neurons that weren't dropped
        return dL_dy * self.mask

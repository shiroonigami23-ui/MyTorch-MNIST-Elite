
class Module:
    def __init__(self):
        # This will store the data from the forward pass
        # that we need during the backward pass (like inputs or intermediate values).
        self.cache = None

    def __call__(self, *args, **kwargs):
        # This allows us to use 'model(x)' instead of 'model.forward(x)'
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        # To be implemented by child classes (Linear, ReLU, etc.)
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        # To be implemented by child classes
        raise NotImplementedError

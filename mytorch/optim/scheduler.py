
class StepLR:
    def __init__(self, optimizer, step_size, gamma=0.5):
        self.optimizer = optimizer
        self.step_size = step_size # Every 'x' epochs
        self.gamma = gamma # Multiply LR by this factor
        self.last_epoch = 0

    def step(self):
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma
            print(f"📉 Learning rate decayed to: {self.optimizer.lr}")

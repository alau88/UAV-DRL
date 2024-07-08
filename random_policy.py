import numpy as np

class RandomPolicyNet:
    def __init__(self, action_space_shape):
        self.action_space_shape = action_space_shape

    def __call__(self, state):
        return np.random.uniform(-1, 1, self.action_space_shape).flatten()

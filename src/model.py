import decoder
import numpy as np


class Perceptron:
    def __init__(self, dim) -> None:
        weight_init = 40

        self.weight = np.random.randint(0, weight_init, dim)

    def forward(self, x):
        h = np.dot(x, self.weight)
        return np.squeeze(h)

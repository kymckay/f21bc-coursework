import numpy as np
from typing import Tuple

act_func: Tuple[function, function]

# Can use Sigmoid function to compress to 0-1 range as probability
def sigmoid(z):
    # Log loss is undefined for probability value of 1 and 0
    # Use very small epsilon to clip result
    eps = 1e-15
    return np.maximum(
        eps, np.minimum(
            1-eps, np.divide(1, 1 + np.exp(-z))
        )
    )

def d_sigmoid(z):
    pass

class layer:
    def __init__(self, nodes: int, func: act_func) -> None:
        self.nodes = nodes
        self.fn, self.der = func

    def input_shape(self, ni: int, nf: int) -> None:
        # Number of instances and features
        self.ni = ni
        self.nf = nf + 1 # bias will be considered an extra input

        # 1 weight per feature
        self.w = np.random.rand(self.nf, 1)


    # values: numpy matrix (instances x features)
    # returns: numpy matrix (instances x nodes)
    def activate(self, values):
        b = np.ones((self.ni, 1))
        x = np.concatenate((values, b), axis=1)

        z = np.matmul(x, self.w)

        # Save output value for use in back prop
        self.y = self.fn(z)

        return self.y

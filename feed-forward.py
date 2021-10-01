import numpy as np
import pandas as pd
from typing import Callable, Tuple

act_func = Tuple[Callable, Callable]

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

        # Weights will be randomly initalised on first activation
        self.first_run = True

    def init_weights(self, shape: Tuple[int,int]) -> None:
        # Number of instances and features
        self.ni, self.nf = shape

        # bias will be considered an extra input
        self.nf += 1

        # 1 weight per feature per node
        self.w = np.random.rand(self.nf, self.nodes)

    # values: numpy matrix (instances x features)
    # returns: numpy matrix (instances x nodes)
    def activate(self, values):
        if self.first_run:
            self.init_weights(values.shape)

        b = np.ones((self.ni, 1))
        x = np.concatenate((values, b), axis=1)

        z = np.matmul(x, self.w)

        # Save output value for use in back prop
        self.y = self.fn(z)

        return self.y


def main():
    l1 = layer(3, (sigmoid, d_sigmoid))

    # The input data (each row is an instance)
    data = pd.read_csv("diabetes.csv", sep=",")
    x = data.loc[:, "Pregnancies":"Age"].to_numpy()

    y = l1.activate(x)
    print(y.shape)

main()
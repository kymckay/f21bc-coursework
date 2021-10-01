import numpy as np
import pandas as pd
from typing import Callable, Iterable, Tuple

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

class act_func:
    def __init__(self, fn: Callable, der: Callable) -> None:
        self.fn = fn
        self.der = der

afuncs = {
    'sigmoid': act_func(sigmoid, d_sigmoid),
}

class layer:
    def __init__(self, nodes: int, act: act_func) -> None:
        self.nodes = nodes
        self.act = act

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
        self.y = self.act.fn(z)

        return self.y

class network:
    # inputs should be a matrix (instances x features)
    # expected_out should be a vector (instances x 1)
    def __init__(self,
        inputs,
        layers: Iterable[layer],
        expected_out
    ) -> None:
        if len(layers) == 0:
            raise ValueError('Need at least one layer in the network')

        self.x = inputs
        self.layers = layers
        self.y = expected_out

    def forward_propagate(self):
        for l, layer in enumerate(self.layers):
            if l == 0:
                y_hat = layer.activate(self.x)
            else:
                # Output is saved in layers
                y_hat = layer.activate(y_hat)
        return y_hat

    def log_loss(self):
        y_hat = self.layers[-1].y

        # Remember these operations are element-wise
        a = np.multiply(self.y, np.log(y_hat))
        b = 1 - self.y
        c = np.log(1 - y_hat)
        return -(a + np.multiply(b,c))

    def backward_propagate(self):
        pass

def main():
    # The input data (each row is an instance)
    data = pd.read_csv("diabetes.csv", sep=",")
    x = data.loc[:, "Pregnancies":"Age"].to_numpy()
    y = data.loc[:, "Outcome"].to_numpy()

    n = network(x, [
        layer(3, afuncs['sigmoid']),
        layer(3, afuncs['sigmoid']),
        layer(1, afuncs['sigmoid'])
    ], y)

    print(n.forward_propagate().shape)

main()
import funcs
import numpy as np
import pandas as pd
from typing import Iterable, Tuple


class layer:
    def __init__(self,
        nodes: int,
        act: funcs.dfunc = funcs.sigmoid
    ) -> None:
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
        expected_out,
        loss_fn: funcs.dfunc = funcs.log_loss
    ) -> None:
        if len(layers) == 0:
            raise ValueError('Need at least one layer in the network')

        self.x = inputs
        self.layers = layers
        self.y = expected_out
        self.loss_fn = loss_fn

    def forward_propagate(self):
        for l, layer in enumerate(self.layers):
            if l == 0:
                y_hat = layer.activate(self.x)
            else:
                # Output is saved in layers
                y_hat = layer.activate(y_hat)
        return y_hat

    def backward_propagate(self):
        pass

def main():
    # The input data (each row is an instance)
    data = pd.read_csv("diabetes.csv", sep=",")
    x = data.loc[:, "Pregnancies":"Age"].to_numpy()
    y = data.loc[:, "Outcome"].to_numpy()

    n = network(x, [
        layer(3),
        layer(3),
        layer(1)
    ], y)

    print(n.forward_propagate().shape)

main()
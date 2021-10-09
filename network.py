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

        # w_jk = weight from input k to node j
        self.w = np.random.rand(self.nodes, self.nf)

    # values: numpy matrix (instances x features)
    # returns: numpy matrix (instances x nodes)
    def activate(self, values):
        if self.first_run:
            self.init_weights(values.shape)

        # Add bias as an extra input to reduce complexity
        b = np.ones((self.ni, 1))

        # Save input values for use in back prop
        self.x = np.concatenate((values, b), axis=1)

        # Save intermediate output for use in back prop
        self.z = self.x @ self.w.T

        return self.act.fn(self.z)

    # alpha is learning rate from NN
    def update_weights(self, dL_dz, alpha: float) -> None:
        # Derivative with respect to w_ij is just x_i
        # So multiply along 2nd axis by x
        dL_dw = dL_dz * self.x

        # Another way of thinking about the above is working back from
        # one dimension of output to one per node (hence, 2D) of input

        # Update the weights using the averaged result across all instances
        # Remember that x-th feature corresponds to x-th connection to each node, so transpose the for adjustment
        self.w = self.w - alpha * np.mean(dL_dw, axis=0).T


class network:
    # inputs should be a matrix (instances x features)
    # expected_out should be a matrix (instances x num_outputs)
    def __init__(self,
        inputs,
        layers: Iterable[layer],
        expected_out,
        alpha = 0.01,
        loss_fn: funcs.dfunc = funcs.log_loss,
    ) -> None:
        if len(layers) == 0:
            raise ValueError('Need at least one layer in the network')
        if (
            expected_out.ndim != 2
            or expected_out.shape[1] != layers[-1].nodes
        ):
            raise ValueError('2nd dimension of exected outputs does not match nodes in output layer')


        self.x = inputs
        self.layers = layers
        self.y = expected_out
        self.alpha = alpha
        self.loss_fn = loss_fn

    def forward_propagate(self):
        for l, layer in enumerate(self.layers):
            if l == 0:
                self.y_hat = layer.activate(self.x)
            else:
                # Each layer saves the input it recieved for back prop
                self.y_hat = layer.activate(self.y_hat)
        return self.y_hat

    def backward_propagate(self):
        # Derivative of loss function, with respect to output y_hat
        # (independent variable will change in back prop via chain rule)
        dL = self.loss_fn.der(self.y, self.y_hat)

        for layer in reversed(self.layers):
            # Derivative with respect to z
            dL_dz = layer.act.der(layer.z) * dL

            # this is dz/dx, column vector
            dz_dx = np.atleast_2d(np.sum(layer.w, axis=1))

            # Drop bias input (no influence on previous layer)
            dz_dx = dz_dx[:, :-1]

            # Repeate derivatives so matrix multiplication aligns
            # (the sum of weights is same for every instance)
            dz_dx = np.repeat(dz_dx, layer.nodes, axis=0)

            # Find derivative with respect to x to chain back a layer
            # Summed acrosses the nodes to reflect influence in multiple places
            dL = dL_dz @ dz_dx

            # Finds derivative respect to w and updates
            layer.update_weights(dL_dz, self.alpha)


def main():
    # The input data (each row is an instance)
    data = pd.read_csv("diabetes.csv", sep=",")
    x = data.loc[:, "Pregnancies":"Age"].to_numpy()
    y = data.loc[:, "Outcome"].to_numpy()[:, np.newaxis]

    n = network(x, [
        layer(1),
        layer(1),
        layer(1)
    ], y)

    n.forward_propagate()
    n.backward_propagate()

main()
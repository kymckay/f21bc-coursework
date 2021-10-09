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
        self.w = np.random.rand(self.nodes, self.nf) * 0.01

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

class network:
    # inputs should be a matrix (instances x features)
    # expected_out should be an array of length (instances)
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
            expected_out.ndim > 1
            or layers[-1].nodes > 1
        ):
            raise ValueError('More than one output not supported')

        self.x = inputs
        self.layers = layers
        self.y = expected_out[:, np.newaxis]
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
        # Derivative of loss function, with respect to
        # activation of current layer
        dL_da = self.loss_fn.der(self.y, self.y_hat)

        for i, layer in enumerate(reversed(self.layers)):
            # First step of chain rule to "unwrap" activation function
            dL_dz = layer.act.der(layer.z) * dL_da

            # Derivative with respect to w_jk is just x_k
            # With multi-nodes, each gets the same input (extra axis)
            dz_dw = np.expand_dims(layer.x, 1)

            # This ends up multiplying broadcasted, going
            # from 1D activation to 2D weight connections
            # (the 3rd dimension (index 0) here for instances
            # already matches)
            # print(dz_dw.shape, dL_dz.shape)
            dL_dw = dz_dw * np.expand_dims(dL_dz, 2)

            # Update the weights using the averaged result
            # across all instances
            layer.w = layer.w - self.alpha * np.mean(dL_dw, axis=0)

            # Once first layer weights update nothing left to do
            if i == len(self.layers) - 1:
                break

            # Derivative with respect to x_k is just w_jk summed over j
            # Drop last k because bias not from previous layer
            dz_dx = np.sum(layer.w[:, :-1], axis=0)[np.newaxis, :]

            # Derivative with respect to activation of
            # previous layer (equivalent to input of this layer)
            dL_da = dz_dx * dL_dz


def main():
    # The input data (each row is an instance)
    data = pd.read_csv("diabetes.csv", sep=",")
    x = data.loc[:, "Pregnancies":"Age"].to_numpy()
    y = data.loc[:, "Outcome"].to_numpy()

    n = network(x, [
        layer(1),
        layer(2),
        layer(2),
        layer(1),
    ], y)

    n.forward_propagate()
    n.backward_propagate()

main()
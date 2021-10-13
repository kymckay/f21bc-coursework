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
    # returns: numpy matrix (instances x neurons)
    def activate(self, values):
        if self.first_run:
            self.init_weights(values.shape)
            self.first_run = False

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
        # activation of current layer.
        # Shape (instances x neurons)
        dL_da = self.loss_fn.der(self.y, self.y_hat)

        # There are three dimensions to consider in all following:
        # instances, neurons, features (all of current layer)
        for i, layer in enumerate(reversed(self.layers)):
            # First step of chain rule to "unwrap" activation function
            # These values are shape (instances x neruons)
            dL_dz = layer.act.der(layer.z) * dL_da

            # Derivative with respect to w_jk is just x_k
            # Hence, the same for all neurons (j=0, ..., n)
            # Shape (instances x features)
            dz_dw = layer.x

            # Matrix multiplication here achieves the first step of
            # averaging across instances (sum of products).
            # Shape (neurons x features)
            dL_dw = dL_dz.T @ dz_dw

            # Dividing by number of instances gets the average
            # Learning rate dictates rate of learning
            layer.w = layer.w - self.alpha * dL_dw / self.x.shape[0]

            # Once input layer weights update, nothing left to do
            if i == len(self.layers) - 1:
                break

            # Derivative with respect to x_k is just w_jk summed over j
            # Drop last k because bias not from previous layer
            # Weights are the same for all instances
            # Shape (neurons x features)
            dz_dx = layer.w[:, :-1]

            # Matrix multiplication here achieves the sum over j
            # mentioned above.
            # Shape (instances x features) equivalent of
            # (instances x neurons in previous layer)
            dL_da = dL_dz @ dz_dx

    def get_loss(self) -> float:
        return np.mean(self.loss_fn.fn(self.y, self.y_hat))

    def learn(self,
        epochs:int = None,
        delta:float = None,
        showProgress:int = 0
    ):
        if (epochs is None) and (delta is None):
            raise ValueError(
                "Must specify epochs to learn over or delta to consider convergance under"
            )

        # Give initial reference point
        self.forward_propagate()
        print(f'0 epochs: loss = {self.get_loss()}')

        if epochs:
            for i in range(1, epochs + 1):
                self.backward_propagate()
                self.forward_propagate()

                if showProgress and (i % showProgress == 0):
                    L = self.get_loss()
                    print(f'{i} epochs: loss = {L}')

        # TODO: delta convergance

def main():
    # The input data (each row is an instance)
    data = pd.read_csv("diabetes.csv", sep=",")
    x = data.loc[:, "Pregnancies":"Age"].to_numpy()
    y = data.loc[:, "Outcome"].to_numpy()

    n = network(x, [
        layer(4),
        layer(4),
        layer(1),
    ], y)

    n.learn(20000, showProgress=1000)

main()
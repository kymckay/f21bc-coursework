import funcs
import numpy as np
from typing import Iterable


class layer:
    def __init__(self,
        n_features: int,
        nodes: int,
        act: funcs.dfunc = funcs.leaky_relu
    ) -> None:
        self.nodes = nodes
        self.act = act

        # bias will be considered an extra feature
        n_features += 1

        # w_jk = weight from feature k to node j
        # initialise weights as Gaussian random variables with mean 0 and SD 1/sqrt(nr_nodes_in_previous_layer)
        self.w = np.array(np.random.randn(self.nodes, n_features)/np.sqrt(n_features))

    # values: numpy matrix (instances x features)
    # returns: numpy matrix (instances x neurons)
    def activate(self, values, remember=False):
        n_instances, _ = values.shape

        # Add bias as an extra feature to reduce complexity
        b = np.ones((n_instances, 1))
        x = np.concatenate((values, b), axis=1)

        # Activation across all instances at once achieved via matrix
        # multiplication
        z = x @ self.w.T

        # Need to recall these values for back propagation in training
        if remember:
            self.x = x
            self.z = z

        # Output dictated by the activation function
        return self.act.fn(z)

class network:
    # Produces a new network from a list representation of properties
    # The network architecture is fixed to 2 hidden layers of 4 nodes
    # List length must be 4 * (n_features + 4 + 2 + 1)
    @staticmethod
    def from_list(props: Iterable[float], n_features: int):
        pass

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
            # Want the layers to remember the input and intermediate
            # results for later back propegation when training
            if l == 0:
                self.y_hat = layer.activate(self.x, remember=True)
            else:
                self.y_hat = layer.activate(self.y_hat, remember=True)
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

    def get_loss(self, expected, probabilities) -> float:
        return np.mean(self.loss_fn.fn(expected, probabilities))

    def train(self, epochs:int = 1, test_x = None, test_y = None):
        # Give initial reference point
        self.forward_propagate()

        # Collection of prediction accuracy for each epoch
        accuracy = []
        loss = []
        accuracy_test = []
        loss_test = []

        if epochs:
            for _ in range(1, epochs + 1):
                self.backward_propagate()
                self.forward_propagate()

                # Predicted output after each epoch
                predicted = np.around(self.y_hat)

                # Accuracy is percentage of correct guesses
                # Equivalent to average here because values are 1 or 0
                accuracy.append(np.mean(predicted == self.y))
                loss.append(self.get_loss(self.y, self.y_hat))

                if (test_x is not None) and (test_y is not None):
                    test_prob = self.test(test_x)
                    test_pred = np.around(test_prob)
                    accuracy_test.append(np.mean(test_pred == test_y))
                    loss_test.append(self.get_loss(test_y, test_prob))

        return (
            np.array(accuracy), np.array(loss),
            np.array(accuracy_test), np.array(loss_test)
        )

    def test(self, input):
        # Just forward propagation that doesn't effect training state
        for l, layer in enumerate(self.layers):
            if l == 0:
                out = layer.activate(input, remember=False)
            else:
                out = layer.activate(out, remember=False)

        # Spit out the probabilities
        return out
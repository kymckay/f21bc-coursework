import numpy as np
from typing import Callable

class dfunc:
    def __init__(self, fn: Callable, der: Callable) -> None:
        self.fn = fn
        self.der = der

# Can use Sigmoid function to compress to 0-1 range as probability
def _sigmoid(z):
    return 1 / (1 + np.exp(-z))

def _d_sigmoid(z):
    return _sigmoid(z) * (1 - _sigmoid(z))

def _log_loss(y, y_hat):
    # Remember these operations are element-wise
    a = y * np.log(y_hat)
    b = 1 - y
    c = np.log(1 - y_hat)
    return -(a + b * c)

def _d_log_loss(y, y_hat):
    # Remember these operations are element-wise
    return -y/y_hat + (1 - y)/(1 - y_hat)

sigmoid = dfunc(_sigmoid, _d_sigmoid)
log_loss = dfunc(_log_loss, _d_log_loss)
import numpy as np
from typing import Callable

class act_func:
    def __init__(self, fn: Callable, der: Callable) -> None:
        self.fn = fn
        self.der = der

# Can use Sigmoid function to compress to 0-1 range as probability
def _sigmoid(z):
    # Log loss is undefined for probability value of 1 and 0
    # Use very small epsilon to clip result
    eps = 1e-15
    return np.maximum(
        eps, np.minimum(
            1-eps, np.divide(1, 1 + np.exp(-z))
        )
    )

def _d_sigmoid(z):
    pass


sigmoid = act_func(_sigmoid, _d_sigmoid)
from numpy import random, ndarray
from typing import Callable


def inverse_transform(*params: tuple, size, ppf: Callable):
    u: ndarray = random.uniform(size=size)
    if params is None:
        return ppf(u)
    return ppf(u, params)


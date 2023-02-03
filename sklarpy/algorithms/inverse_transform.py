# Contains the inverse transformation method for generating pseudo-random numbers from a probability distribution.
from numpy import random, ndarray
from typing import Callable

__all__ = ['inverse_transform']


def inverse_transform(*params: tuple, size, ppf: Callable):
    """
    Generates a random number from a probability distribution.

    Parameters
    ----------
    params: tuple
        The parameters specifying the probability distribution.
    size:
        The size/shape of the generated array containing random numbers from your distribution.
    ppf: Callable
        The inverse function of the discrete distribution. Must take a numpy array containing quantile values and the
        distribution's parameters in a tuple as arguments.
    """
    u: ndarray = random.uniform(size=size)
    if params is None:
        return ppf(u)
    return ppf(u, params)

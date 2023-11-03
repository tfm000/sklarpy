# Contains code implementing the inverse transformation method for generating
# pseudo-random numbers from a univariate probability distribution.
from numpy import random, ndarray
from typing import Callable

__all__ = ['inverse_transform']


def inverse_transform(*params: tuple, size, ppf: Callable, **kwargs):
    """Generates a pseudo random sample of a univariate probability
    distribution.

    Parameters
    ----------
    params: tuple
        The parameters specifying the univariate probability distribution.
    size:
        The size/shape of the generated array containing random numbers from
        the univariate distribution.
    ppf: Callable
        The inverse of the cumulative density function distribution of the
        univariate distribution. Must take a numpy array containing [0-1]
        values and the distribution's parameters in a tuple as arguments.
    kwargs:
        Any additional keyword arguments to pass to the ppf function.

    Returns
    -------
    rvs:
        Randomly sampled pseudo observations.
    """
    shape: tuple = (size, ) if isinstance(size, int) else size
    u: ndarray = random.uniform(size=size)
    vals = ppf(u, **kwargs) if params is None else ppf(u, params, **kwargs)
    return vals.reshape(shape)

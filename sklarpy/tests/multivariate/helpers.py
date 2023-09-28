# Contains helper functions for testing SklarPy multivariate code
import os

from sklarpy.multivariate import *
from sklarpy import load

__all__ = ['get_dist']


def get_dist(name: str, d: int = 3) -> tuple:
    """Fits a distribution to data.

    Parameters
    ----------
    name : str
        The name of the distribution as in distributions_map
    d: int
        The dimension of the data.
    Returns
    -------
    res: tuple
        non-fitted dist, fitted dist, parameters for fitted dist
    """
    dist = eval(name)
    path: str = f'{os.getcwd()}\\sklarpy\\tests\\multivariate\\{name}_{d}D'
    params = load(path)
    fitted = dist.fit(params=params)
    return dist, fitted, params

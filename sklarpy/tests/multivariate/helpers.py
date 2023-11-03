# Contains helper functions for testing SklarPy multivariate code
import numpy as np

from sklarpy.multivariate import *

__all__ = ['get_dist']


def get_dist(name: str, params_dict: dict, data: np.ndarray = None) -> tuple:
    """Fits a distribution to data.

    Parameters
    ----------
    name : str
        The name of the distribution as in distributions_map.
    params_dict: dict
        Dictionary containing the parameters of each distribution.
    data: np.ndarray
        Multivariate data to fit the distribution too if its params are not
        specified in params_dict.

    Returns
    -------
    res: tuple
        non-fitted dist, fitted dist, parameters for fitted dist
    """
    dist = eval(name)
    if name in params_dict:
        params: tuple = params_dict[name]
        fitted = dist.fit(params=params)
    else:
        fitted = dist.fit(data=data)
        params: tuple = fitted.params.to_tuple
    return dist, fitted, params

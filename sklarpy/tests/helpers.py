# Helper functions for tests
import numpy as np

from sklarpy.univariate import *


def get_data(name: str, uniform_data: np.ndarray, poisson_data: np.ndarray, extension='') -> np.ndarray:
    """contains logic for selecting the correct data for a given distribution.

    Parameters
    ==========
    name : str
        The name of the distribution as in distributions_map
    uniform_data: np.ndarray
        Random numbers generated from a uniform distribution
    poisson_data: np.ndarray
        Random numbers generated from a poisson distribution
    extension : str
        'numerical', 'parametric' or ''

    Returns
    ========
    data : np.ndarray
        The data to fit the distribution too.
    """
    if name in distributions_map[f'all continuous{extension}']:
        # continuous data
        return uniform_data
    else:
        # discrete data
        return poisson_data


def get_dist(name: str, data: np.ndarray) -> tuple:
    """Fits a distribution to data.

    Parameters
    ===========
    name : str
        The name of the distribution as in distributions_map
    data: np.ndarray
        The data to fit the distribution too

    Returns
    =======
    res: tuple
        non-fitted dist, fitted dist, parameters for fitted dist
    """
    dist = eval(name)
    fitted = dist.fit(data)
    params: tuple = fitted.params
    return dist, fitted, params


def get_fitted_dict(name: str, data: np.ndarray) -> dict:
    """Fits distribution to data and (if parametric) to parameters.

    Parameters
    ===========
    name : str
        The name of the distribution as in distributions_map
    data: np.ndarray
        The data to fit the distribution too

    Returns
    =======
    fitted_dict: dict
        Dictionary containing fitted distributions.
    """
    dist = eval(name)
    data_fitted = dist.fit(data)
    fitted_dict: dict = {'fitted to data': data_fitted}

    if name in distributions_map['all numerical']:
        return fitted_dict

    params: tuple = data_fitted.params
    params_fitted = dist.fit(params=params)
    fitted_dict['fitted to params'] = params_fitted
    return fitted_dict

# Contains helper functions for testing SklarPy univariate code
import numpy as np

from sklarpy.univariate import *
from sklarpy.univariate._fitted_dists import FittedContinuousUnivariate, \
    FittedDiscreteUnivariate

__all__ = ['get_data', 'get_target_fit', 'get_dist']


def get_data(name: str, continuous_data: np.ndarray, discrete_data: np.ndarray,
             ) -> np.ndarray:
    """contains logic for selecting the correct data for a given distribution.

    Parameters
    ---------
    name : str
        The name of the distribution as in distributions_map
    continuous_data: np.ndarray
        Random numbers generated from a continuous distribution
    discrete_data: np.ndarray
        Random numbers generated from a discrete distribution

    Returns
    -------
    data : np.ndarray
        The data to fit the distribution too.
    """
    if name in distributions_map[f'all continuous']:
        # continuous data
        return continuous_data
    else:
        # discrete data
        return discrete_data


def get_target_fit(name: str, continuous_data: np.ndarray,
             discrete_data: np.ndarray,
             ) -> np.ndarray:
    """contains logic for selecting the correct data for a given distribution.

    Parameters
    ---------
    name : str
        The name of the distribution as in distributions_map
    continuous_data: np.ndarray
        Random numbers generated from a continuous distribution
    discrete_data: np.ndarray
        Random numbers generated from a discrete distribution

    Returns
    -------
    target_fit : np.ndarray
        The fitted distribution object target.
    """
    if name in distributions_map[f'all continuous']:
        # continuous distribution
        return FittedContinuousUnivariate
    else:
        # discrete distribution
        return FittedDiscreteUnivariate


def get_dist(name: str, data: np.ndarray) -> tuple:
    """Fits a distribution to data.

    Parameters
    ----------
    name : str
        The name of the distribution as in distributions_map
    data: np.ndarray
        The data to fit the distribution too

    Returns
    -------
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
    -----------
    name : str
        The name of the distribution as in distributions_map
    data: np.ndarray
        The data to fit the distribution too

    Returns
    -------
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

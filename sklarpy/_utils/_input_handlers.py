# Contains functions for handling user inputs
import numpy as np
from pandas import Series, DataFrame
from typing import Iterable

from sklarpy._utils._variable_types import num_or_array, data_iterable

__all__ = ['univariate_num_to_array', 'check_params', 'check_univariate_data', 'check_array_datatype',
           'check_multivariate_data']


def univariate_num_to_array(x: num_or_array) -> np.ndarray:
    """
    Convert a number or numpy array to a flattened numpy array.

    Parameters
    ----------
    x: num_or_array
        The number or array to convert.

    Returns
    -------
    numpy.ndarray
        The numpy array.
    """
    if isinstance(x, np.ndarray):
        return x.flatten()
    elif isinstance(x, float) or isinstance(x, int):
        return np.asarray(x).flatten()
    raise TypeError("input must be a numpy array, float or integer")


def check_params(params: tuple) -> tuple:
    """
    Checks whether the given parameters match those required by SklarPy's probability distributions.

    Parameters
    ----------
    params: tuple
        The parameters of the probability distribution.

    Returns
    -------
    tuple
        The parameters of the probability distribution.
    """
    if isinstance(params, tuple):
        return params
    raise TypeError("params must be a tuple")


def check_univariate_data(data: data_iterable) -> np.ndarray:
    """
    Checks user inputted data for univariate distribution fitting.

    Parameters
    ----------
    data: data_iterable
        The data to check.

    Returns
    -------
    data: np.ndarray
        The input data converted into a flattened numpy array.
    """
    if isinstance(data, np.ndarray):
        return data.flatten()
    elif isinstance(data, DataFrame):
        if len(data.columns) != 1:
            raise ValueError("data must be a single column dataframe for it to be considered univariate.")
        return data.to_numpy()
    elif isinstance(data, Series):
        return data.to_numpy()
    elif isinstance(data, Iterable):
        return np.asarray(data).flatten()
    else:
        raise TypeError("data must be an iterable.")


def check_array_datatype(arr: np.ndarray, must_be_numeric: bool = True):
    """
    Checks the data-type of numpy arrays.

    Parameters
    ----------
    arr: np.ndarray
        The numpy array whose data-type must be determined.
    must_be_numeric: bool
        Whether the data-type must be numeric (float or integer) or not.

    Returns
    -------
    data-type
        the data-type of the numpy array.
    """
    if not ((arr.dtype == float) or (arr.dtype == int)):
        if must_be_numeric:
            raise TypeError("Array must contain integer or float values.")
        return arr.dtype

    arr_int: np.ndarray = arr.astype(int)
    if np.all((arr - arr_int) == 0):
        return int
    return float


def check_multivariate_data(data: data_iterable, num_variables: int = None, allow_1d: bool = False, allow_nans: bool = True) -> np.ndarray:
    # converting to numpy array
    data_array: np.ndarray = np.asarray(data)

    # checking dimensions
    data_shape = data_array.shape
    if len(data_shape) == 1 and allow_1d:
        data_array = data_array.reshape((1, data_shape[0]))
    elif len(data_shape) != 2:
        raise ValueError("data must be 2-dimensional.")

    # checking data contains only numbers
    if not ((data_array.dtype == float) or (data_array.dtype == int)):
        raise ValueError("data must only contain integers or floats.")

    # checking number of variables
    if num_variables is not None:
        if data_array.ndim != num_variables:
            raise ValueError("data dimensions do not match the number of variables.")

    # checking for nan values
    if not (allow_nans and np.isnan(data_array).sum() == 0):
        raise ValueError("data must not contain NaN values")
    return data_array

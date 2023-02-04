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


def check_multivariate_data(data: data_iterable, num_variables: int = None) -> tuple:
    """
    Checks user inputted data for multivariate distribution fitting.

    Parameters
    ----------
    data: data_iterable
        The data to check.

    Returns
    -------
    tuple: tuple
        Data converted into a 2D numpy array, dictionary with columns as keys and data indexes as values
    """
    # getting column names if possible
    columns_indices: dict = None
    if isinstance(data, DataFrame):
        columns_indices = {column: i for i, column in enumerate(data.columns)}
    elif isinstance(data, Series):
        columns_indices = {data.name: 0}

    # converting to numpy array
    data: np.ndarray = np.asarray(data)
    if columns_indices is None:
        data_shape = data.shape
        if len(data_shape) != 2:
            raise ValueError("data must be 2-dimensional.")
        columns_indices = {i: i for i in range(data_shape[1]) }
    if len(columns_indices) < 2:
        raise ValueError("data must be 2-dimensional.")

    # checking data contains only numbers
    if not ((data.dtype == float) or (data.dtype == int)):
        raise ValueError("data must only contain integers or floats.")

    # checking number of variables
    if num_variables is not None:
        if len(columns_indices) != num_variables:
            raise ValueError("data dimensions do not match the number of variables.")

    return data, columns_indices

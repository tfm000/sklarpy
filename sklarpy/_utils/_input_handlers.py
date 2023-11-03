# Contains functions for handling user inputs
import numpy as np
import pandas as pd
from typing import Iterable, Union

__all__ = ['univariate_num_to_array', 'check_params', 'check_univariate_data',
           'check_array_datatype', 'check_multivariate_data', 'get_mask']


def univariate_num_to_array(x: Union[float, int, np.ndarray]) -> np.ndarray:
    """
    Convert a number or numpy array to a flattened numpy array.

    Parameters
    ----------
    x: Union[float, int, np.ndarray]
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
    Checks whether the given parameters match those required by SklarPy's
    probability distributions.

    Parameters
    ----------
    params: tuple
        The parameters of the probability distribution.

    Returns
    -------
    tuple
        The parameters of the probability distribution.
    """
    if not isinstance(params, tuple):
        raise TypeError("params must be a tuple")

    for param in params:
        if not (isinstance(param, int) or isinstance(param, float)):
            raise ValueError("all parameters must be scalar values.")

    return params


def check_univariate_data(
        data: Union[pd.DataFrame, pd.Series, np.ndarray, Iterable]) \
        -> np.ndarray:
    """
    Checks user inputted data for univariate distribution fitting.

    Parameters
    ----------
    data: Union[pd.DataFrame, pd.Series, np.ndarray, Iterable]
        The data to check.

    Returns
    -------
    data: np.ndarray
        The input data converted into a flattened numpy array.
    """
    if isinstance(data, pd.Series):
        data = data.to_frame()
    if isinstance(data, pd.DataFrame):
        if len(data.columns) != 1:
            raise ValueError("data must be a single column dataframe for it to"
                             " be considered univariate.")
        data_array: np.ndarray = data.to_numpy().flatten()
    elif isinstance(data, Iterable):
        return np.asarray(list(data)).flatten()
    else:
        raise TypeError("data must be an iterable.")

    # checking data contains only numbers
    dtype = data_array.dtype
    if not ((dtype == float) or (dtype == int)
            or (dtype == np.int32) or (dtype == np.int64)
            or (dtype == np.float32) or (dtype == np.float64)):
        raise ValueError("data must only contain integers or floats.")

    return data_array


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
    dtype = arr.dtype
    if not ((dtype == float) or (dtype == int)
            or (dtype == np.int32) or (dtype == np.int64)
            or (dtype == np.float32) or (dtype == np.float64)):
        if must_be_numeric:
            raise TypeError("Array must contain integer or float values.")
        return arr.dtype

    arr_int: np.ndarray = arr.astype(int)
    if np.all((arr - arr_int) == 0):
        return int
    return float


def check_multivariate_data(
        data: Union[pd.DataFrame, pd.Series, np.ndarray, Iterable],
        num_variables: int = None, allow_1d: bool = False,
        allow_nans: bool = True) -> np.ndarray:
    """Checks user inputted data for multivariate distribution fitting.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series, np.ndarray, Iterable]
        The data to check.
    num_variables: int
        Optional. The required number of variables.
        If the number of variables is flexible / does not need to be specified,
        parse None.
        Default is None.
    allow_1d: bool
        Optional. Whether to allow users to input 1d data.
        If True, the data will be transformed into 2d arrays with shape (n, 2),
        where n is the number of datapoints in the original data.
        Default is False.
    allow_nans: bool
        Optional. True to allow nans in the dataset.
        Default is True

    Returns
    -------
    data_array: np.ndarray
        The user's dataset, transformed into a 2d numpy array.
    """
    # converting to numpy array
    data_array: np.ndarray = np.asarray(data)

    # checking dimensions
    data_shape = data_array.shape
    if len(data_shape) == 1 and allow_1d:
        data_array = data_array.reshape((1, data_shape[0]))
    elif len(data_shape) != 2:
        raise ValueError("data must be 2-dimensional.")

    # checking data contains only numbers
    dtype = data_array.dtype
    if not ((dtype == float) or (dtype == int)
            or (dtype == np.int32) or (dtype == np.int64)
            or (dtype == np.float32) or (dtype == np.float64)):
        raise ValueError("data must only contain integers or floats.")

    # checking number of variables
    if num_variables is not None:
        if data_array.ndim != num_variables:
            raise ValueError("data dimensions do not match the number of "
                             "variables.")

    # checking for nan values
    if (not allow_nans) and (np.isnan(data_array).sum() != 0):
        raise ValueError("data must not contain NaN values")
    return data_array


def get_mask(data: np.ndarray) -> tuple:
    """Creates a mask for nan values in the given dataset.

    Parameters
    ----------
    data: np.ndarray
        Numpy array which potentially contains NaN values.

    Returns
    -------
    output: tuple
        mask array, data with mask applied, array of nan values the same shape
        as the original data.
    """
    output: np.ndarray = np.full((data.shape[0], ), np.nan)
    mask: np.ndarray = np.isnan(data).any(axis=1)
    return mask, data[~mask], output

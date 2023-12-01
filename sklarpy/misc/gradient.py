# contains code for calculating gradients of functions
import numpy as np
import pandas as pd
from typing import Callable, Union, Iterable
from collections import deque

from sklarpy.utils._input_handlers import check_univariate_data

__all__ = ['gradient_1d']


def gradient_1d(func: Callable,
                x: Union[pd.DataFrame, pd.Series, np.ndarray, Iterable],
                eps: float = 10 ** -2,  domain: tuple = None) -> np.ndarray:
    """Calculates the numerical first derivative / gradient of a given
    1-dimensional function.

    Parameters
    ----------
    func : Callable
        The function to differentiate. Must accept scalar values as arguments.
    x: Union[pd.DataFrame, pd.Series, np.ndarray, Iterable]
        The data points to calculate the derivative at. Must be a 1-dimensional
        array, dataframe, series or iterable containing integer or scalar
        values.
    eps: float
        The epsilon value to use when computing the numerical first derivative.
        Default is 0.01.
    domain: tuple
        The domain on which your function is valid. Optional.

    Returns
    --------
    gradient : ndarray
        The numerical gradient of your function.
    """
    # checking arguments
    x: np.ndarray = check_univariate_data(x)

    if domain is None:
        domain = x.min(), x.max()
    elif (not isinstance(domain, tuple)) or (len(domain) != 2):
        raise TypeError('domain must be None or a length 2 tuple.')
    else:
        domain = min(domain), max(domain)

    if (not isinstance(eps, float)) or (eps <= 0):
        raise TypeError('eps must be a positive float.')

    if not isinstance(func, Callable):
        raise TypeError('func must be a callable function.')

    # calculating gradients
    gradients: deque = deque()
    for xi in x:
        if (xi < domain[0]) or (xi > domain[1]):
            # nan if outside domain
            gradients.append(np.nan)
        else:
            # determining upper and lower values to use when calculating the
            # derivative
            if xi - eps >= domain[0]:
                lower: float = xi - eps
            else:
                lower = xi

            if xi + eps <= domain[1]:
                upper: float = xi + eps
            else:
                upper = xi

            # calculating numerical derivative
            gradient: float = (func(upper) - func(lower)) / (upper - lower)
            gradients.append(gradient)
    return np.asarray(gradients)

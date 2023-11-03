# Contains code for fitting a continuous empirical distribution
import numpy as np
from collections import deque
import scipy.interpolate
from typing import Callable
from functools import partial

from sklarpy.univariate._distributions._numerical_wrappers import \
    NumericalWrappers
from sklarpy.misc import gradient_1d

__all__ = ['continuous_empirical_fit']


def continuous_empirical_fit(data: np.ndarray) -> tuple:
    """Fitting function for a univariate continuous empirical distribution.

    Parameters
    -----------
    data : np.ndarray
        The data to fit to the continuous empirical distribution in a
        flattened numpy array.

    Returns
    --------
    fitted_funcs: tuple
        fitted pdf, cdf, ppf, support, rvs functions
    """
    num_data_points: int = data.size
    xmin, xmax = data.min(), data.max()
    sorted_data: np.ndarray = np.sort(data)

    # calculating empirical cdf
    empirical_cdf_values = deque()
    for x in sorted_data:
        p: float = (sorted_data <= x).sum() / num_data_points
        empirical_cdf_values.append(p)
    empirical_cdf_values: np.ndarray = np.asarray(empirical_cdf_values)
    cdf_: Callable = scipy.interpolate.interp1d(
        sorted_data, empirical_cdf_values, 'linear', bounds_error=False
    )
    cdf: Callable = partial(
        NumericalWrappers.numerical_cdf, cdf_=cdf_, xmin=xmin, xmax=xmax
    )

    # calculating empirical pdf
    empirical_pdf_values: np.ndarray = gradient_1d(cdf, sorted_data)
    pdf_: Callable = scipy.interpolate.interp1d(
        sorted_data, empirical_pdf_values, 'linear', bounds_error=False
    )
    pdf: Callable = partial(NumericalWrappers.numerical_pdf, pdf_=pdf_)

    # calculating empirical ppf
    F_xmin, F_xmax = cdf(np.array([xmin, xmax]))
    ppf_: Callable = scipy.interpolate.interp1d(
        empirical_cdf_values, sorted_data, 'linear', bounds_error=False
    )
    ppf: Callable = partial(
        NumericalWrappers.numerical_ppf, ppf_=ppf_, xmin=xmin,
        xmax=xmax, F_xmin=F_xmin, F_xmax=F_xmax
    )

    # empirical support
    support: Callable = partial(
        NumericalWrappers.numerical_support, xmin=xmin, xmax=xmax
    )

    return pdf, cdf, ppf, support, None

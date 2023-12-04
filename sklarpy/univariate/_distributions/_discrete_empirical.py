# Contains code for fitting a discrete empirical distribution
import numpy as np
from typing import Callable
from functools import partial
import scipy.interpolate

from sklarpy.univariate._distributions._numerical_wrappers import \
    NumericalWrappers
from sklarpy.utils._errors import DiscreteError

__all__ = ['discrete_empirical_fit']


def discrete_empirical_pdf(x, data, N):
    """the pdf function for a discrete empirical distribution."""
    return np.array([np.count_nonzero(data == i) / N for i in x])


def discrete_empirical_fit(data: np.ndarray) -> tuple:
    """Fitting function for a univariate discrete empirical distribution.

    Parameters
    ----------
    data : np.ndarray
        The data to fit to the discrete empirical distribution in a flattened
        numpy array.

    Returns
    -------
    fitted_funcs: tuple
        fitted pdf, cdf, ppf, support, rvs functions
    """
    xmin, xmax = data.min(), data.max()
    num_data_points: int = data.size

    # Fitting pdf function
    pdf_: Callable = partial(discrete_empirical_pdf, data=data,
                             N=num_data_points)
    pdf: Callable = partial(NumericalWrappers.numerical_pdf, pdf_=pdf_)

    # Generating cdf and ppf functions via interpolation
    empirical_range: np.ndarray = np.arange(xmin, xmax + 1, dtype=int)
    empirical_cdf: np.ndarray = np.cumsum(pdf(empirical_range))
    _, idx = np.unique(empirical_cdf, return_index=True)
    empirical_range, empirical_cdf = empirical_range[idx], empirical_cdf[idx]

    if (empirical_cdf.size == 1) and (empirical_cdf[0] == 0):
        raise DiscreteError("Discrete empirical distribution cannot be fit on "
                            "continuous data.")

    # Fitting cdf, ppf and support functions
    cdf_: Callable = scipy.interpolate.interp1d(
        empirical_range, empirical_cdf, 'nearest', bounds_error=False
    )
    cdf: Callable = partial(
        NumericalWrappers.numerical_cdf, cdf_=cdf_, xmin=xmin, xmax=xmax
    )
    F_xmin, F_xmax = cdf(np.array([xmin, xmax]))
    ppf_: Callable = scipy.interpolate.interp1d(
        empirical_cdf, empirical_range, 'nearest', bounds_error=False
    )
    ppf: Callable = partial(
        NumericalWrappers.numerical_ppf, ppf_=ppf_, xmin=xmin,
        xmax=xmax, F_xmin=F_xmin, F_xmax=F_xmax
    )
    support: Callable = partial(
        NumericalWrappers.numerical_support, xmin=xmin, xmax=xmax
    )

    return pdf, cdf, ppf, support, None

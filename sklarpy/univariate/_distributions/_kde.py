# Contains code for fitting a gaussian kernel density function as a
# probability distribution
import numpy as np
import scipy.stats
from typing import Callable
from functools import partial
import scipy.interpolate

from sklarpy.univariate._distributions._numerical_wrappers import \
    NumericalWrappers

__all__ = ['kde_fit']


def kde_cdf(x, kde):
    """calculates the cdf of a fitted gaussian kde"""
    return np.array([kde.integrate_box_1d(-np.inf, v) for v in x])


def kde_rvs(kde, size: tuple):
    """A function used to ensure the output of the gaussian_kde rvs function
    is correct.

    Parameters
    ----------
    kde:
        A scipy.stats.gaussian_kde fitted to data
    size: tuple
        The size/dimensions of the random variables to generate.

    Returns
    -------
    rvs: np.ndarray
        rvs
    """
    num_to_generate: int = 1
    for dim in size:
        num_to_generate *= dim
    return kde.resample(num_to_generate).reshape(size)


def kde_fit(data: np.ndarray) -> tuple:
    """Fitting function for a univariate gaussian kernel density estimator
    distribution.

    Parameters
    ----------
    data : np.ndarray
        The data to fit to the gaussian kde distribution in a flattened numpy
        array.

    Returns
    -------
    fitted_funcs: tuple
        fitted pdf, cdf, ppf, support, rvs functions
    """
    xmin, xmax = data.min(), data.max()

    kde = scipy.stats.gaussian_kde(data)
    cdf_: Callable = partial(kde_cdf, kde=kde)

    # fitting our distribution functions
    pdf: Callable = partial(NumericalWrappers.numerical_pdf, pdf_=kde.pdf)
    cdf: Callable = partial(NumericalWrappers.numerical_cdf, cdf_=cdf_,
                            xmin=xmin, xmax=xmax)

    F_xmin, F_xmax = cdf(np.array([xmin, xmax]))
    empirical_range: np.ndarray = np.linspace(xmin, xmax, 100)
    ppf_: Callable = scipy.interpolate.interp1d(
        cdf(empirical_range), empirical_range, 'linear', bounds_error=False
    )
    ppf: Callable = partial(
        NumericalWrappers.numerical_ppf, ppf_=ppf_, xmin=xmin,
        xmax=xmax, F_xmin=F_xmin, F_xmax=F_xmax
    )
    support: Callable = partial(
        NumericalWrappers.numerical_support, xmin=xmin, xmax=xmax
    )
    rvs: Callable = partial(kde_rvs, kde=kde)

    return pdf, cdf, ppf, support, rvs

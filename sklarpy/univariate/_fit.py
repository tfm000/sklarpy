# Contains fitting functions for univariate distributions.
import math
import numpy as np
import scipy.interpolate
import scipy.stats
from typing import Callable
from functools import partial
from pandas import isna

from sklarpy._utils import DiscreteError

__all__ = ['dlaplace_fit', 'duniform_fit', 'geometric_fit', 'planck_fit', 'poisson_fit', 'kde_fit',
           'discrete_empirical_fit', 'continuous_empirical_fit']


########################################################################################################################
# Parametric
########################################################################################################################
def dlaplace_fit(data: np.ndarray) -> tuple:
    """Fitting function for the discrete laplacian distribution. Returns the MLE estimator of the parameter alpha.

    Math
    -----
    alpha_mle = arsinh( n/sum(|x_i|) )

    See Also:
    ---------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dlaplace.html#scipy.stats.dlaplace

    Parameters
    ===========
    data : np.ndarray
        The data to fit to the discrete laplace distribution in a flattened numpy array.

    Returns
    =======
    mle_estimator: tuple
       (alpha, )
    """
    return math.asinh(len(data) / abs(data).sum()),


def duniform_fit(data: np.ndarray) -> tuple:
    """Fitting function for the discrete uniform distribution. Returns the minimum and maximum values of the sample.

    Math
    -----
    a = min(x)
    b = max(x)

    See Also:
    ---------
    https://en.wikipedia.org/wiki/Discrete_uniform_distribution

    Parameters
    ===========
    data: np.ndarray
        The data to fit to the discrete uniform distribution in a flattened numpy array.

    Returns
    =======
    estimator: tuple
       (a, b)
    """
    return data.min(), data.max() + 1


def geometric_fit(data: np.ndarray) -> tuple:
    """Fitting function for the geometric distribution. Returns the MLE estimator of the parameter p.

    Math
    -----
    p_mle = n / sum(x)

    See Also:
    ---------
    https://en.wikipedia.org/wiki/Geometric_distribution

    Parameters
    ===========
    data : np.ndarray
        The data to fit to the geometric distribution in a flattened numpy array.

    Returns
    =======
    mle_estimator: tuple
       (p, )
    """
    return len(data) / sum(data),


def planck_fit(data: np.ndarray) -> tuple:
    """Fitting function for the planck exponential distribution. Returns the MLE estimator of the parameter lambda.

    Math
    -----
    lambda_mle = ln(1 + n/sum(x))

    See Also:
    ---------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.planck.html#scipy.stats.planck

    Parameters
    ===========
    data : np.ndarray
        The data to fit to the planck exponential distribution in a flattened numpy array.

    Returns
    =======
    mle_estimator: tuple
       (lambda, )
    """
    return np.log(1 + (len(data) / data.sum())),


def poisson_fit(data: np.ndarray) -> tuple:
    """Fitting function for the poisson distribution. Returns the MLE estimator of the parameter lambda.

    Math
    -----
    lambda_mle = sum(x) / n

    See Also:
    ---------
    https://en.wikipedia.org/wiki/Poisson_distribution

    Parameters
    ===========
    data : np.ndarray
        The data to fit to the poisson distribution in a flattened numpy array.

    Returns
    =======
    mle_estimator: tuple
       (lambda, )
    """
    return data.mean(),


########################################################################################################################
# Numerical/Non-Parametric
########################################################################################################################
def numerical_pdf(x: np.ndarray, pdf_: Callable) -> np.ndarray:
    """A function used to ensure the outputs of a numerical/empirical pdf function are valid

    Parameters
    ===========
    x : np.ndarray
        The values to be parsed into a numerical pdf distribution function.
    pdf_: Callable
        A function interpolating between the data and its empirical pdf values.

    Returns
    =======
    pdf_values : np.ndarray
        An array of valid pdf values.
    """
    raw_pdf_values: np.ndarray = pdf_(x)
    pdf_values: np.ndarray = np.where(~isna(raw_pdf_values), raw_pdf_values, 0.0)
    return np.clip(pdf_values, 0.0, 1.0)


def numerical_cdf(x, cdf_: Callable, xmin, xmax) -> np.ndarray:
    """A function used to ensure the outputs of a numerical/empirical cdf function are valid

    Parameters
    ===========
    x : np.ndarray
        The values to be parsed into a numerical cdf distribution function.
    cdf_: Callable
        A function interpolating between the data and its empirical cdf values.
    xmin:
        The minimum value of our dataset
    xmax:
        The maximum value of our dataset

    Returns
    =======
    cdf_values : np.ndarray
        An array of valid cdf values.
    """
    raw_cdf_values: np.ndarray = cdf_(x)
    cdf_values: np.ndarray = np.where(x >= xmin, raw_cdf_values, 0.0)
    cdf_values = np.where(x <= xmax, cdf_values, 1.0)
    return np.clip(cdf_values, 0.0, 1.0)


def numerical_ppf(x, ppf_: Callable, xmin, xmax, F_xmin: float, F_xmax: float) -> np.ndarray:
    """A function used to ensure the outputs of a numerical/empirical ppf function are valid

        Parameters
        ===========
        x : np.ndarray
            The values to be parsed into a numerical cdf distribution function.
        ppf_: Callable
            A function interpolating between the data and its empirical ppf values.
        xmin:
            The minimum value of our dataset
        xmax:
            The maximum value of our dataset
        F_xmin: float
            The value of the cdf function evaluated at xmin
        F_xmax: float
            The value of the cdf function evaluated at xmax

        Returns
        =======
        ppf_values : np.ndarray
            An array of valid ppf values.
        """
    raw_ppf_values: np.ndarray = ppf_(x)
    ppf_values: np.ndarray = np.where(x >= F_xmin, raw_ppf_values, xmin)
    ppf_values = np.where(x <= F_xmax, ppf_values, xmax)
    return ppf_values


def numerical_support(xmin, xmax) -> tuple:
    """A function which returns the min and max values of the support of a numerical distribution.

    Parameters
    ===========
    xmin:
        The minimum value of our dataset
    xmax:
        The maximum value of our dataset

    Returns
    ========
    support: tuple
        xmin, xmax
    """
    return xmin, xmax


def kde_rvs(kde, size: tuple):
    """A function used to ensure the output of the gaussian_kde rvs function is correct.
    Parameters
    ==========
    kde:
        A scipy.stats.gaussian_kde fitted to data
    size: tuple
        The size/dimensions of the random variables to generate.

    Returns
    =======
    rvs: np.ndarray
        rvs
    """
    num_to_generate: int = 1
    for dim in size:
        num_to_generate *= dim
    return kde.resample(num_to_generate).reshape(size)


def kde_fit(data: np.ndarray) -> tuple:
    """Fitting function for a univariate gaussian kernel density estimator distribution.

    Parameters
    ===========
    data : np.ndarray
        The data to fit to the gaussian kde distribution in a flattened numpy array.

    Returns
    =======
    fitted_funcs: tuple
        fitted pdf, cdf, ppf, support, rvs functions
    """
    interpolation_method: str = 'linear'

    xmin, xmax = data.min(), data.max()

    kde = scipy.stats.gaussian_kde(data)

    # fitting our distribution functions
    pdf_: Callable = kde.pdf
    cdf_: Callable = (lambda x: np.array([kde.integrate_box_1d(-np.inf, v) for v in x]))
    pdf: Callable = partial(numerical_pdf, pdf_=pdf_)
    cdf: Callable = partial(numerical_cdf, cdf_=cdf_, xmin=xmin, xmax=xmax)

    F_xmin, F_xmax = cdf(np.array([xmin, xmax]))
    empirical_range: np.ndarray = np.linspace(xmin, xmax, 100)
    ppf_: Callable = scipy.interpolate.interp1d(cdf(empirical_range), empirical_range, interpolation_method,
                                               bounds_error=False)
    ppf: Callable = partial(numerical_ppf, ppf_=ppf_, xmin=xmin, xmax=xmax, F_xmin=F_xmin, F_xmax=F_xmax)
    support: Callable = partial(numerical_support, xmin=xmin, xmax=xmax)
    rvs: Callable = partial(kde_rvs, kde=kde)

    return pdf, cdf, ppf, support, rvs


def discrete_empirical_pdf(x, data, N):
    """the pdf function for a discrete empirical distribution.
    """
    return np.array([np.count_nonzero(data == i) / N for i in x])


def discrete_empirical_fit(data: np.ndarray) -> tuple:
    """Fitting function for a univariate discrete empirical distribution.

    Parameters
    ===========
    data : np.ndarray
        The data to fit to the discrete empirical distribution in a flattened numpy array.

    Returns
    =======
    fitted_funcs: tuple
        fitted pdf, cdf, ppf, support, rvs functions
    """
    interpolation_method: str = 'nearest'

    xmin, xmax = data.min(), data.max()
    N: int = len(data)

    # Fitting pdf function
    pdf_: Callable = partial(discrete_empirical_pdf, data=data, N=N)
    pdf: Callable = partial(numerical_pdf, pdf_=pdf_)

    # Generating cdf and ppf functions via interpolation
    empirical_range: np.ndarray = np.arange(xmin, xmax + 1, dtype=int)
    empirical_cdf: np.ndarray = np.cumsum(pdf(empirical_range))
    _, idx = np.unique(empirical_cdf, return_index=True)
    empirical_range, empirical_cdf = empirical_range[idx], empirical_cdf[idx]

    if (empirical_cdf.size == 1) and (empirical_cdf[0] == 0):
        raise DiscreteError("Discrete empirical distribution cannot be fit on continuous data.")

    # Fitting cdf, ppf and support functions
    cdf_: Callable = scipy.interpolate.interp1d(empirical_range, empirical_cdf, interpolation_method,
                                                bounds_error=False)
    cdf: Callable = partial(numerical_cdf, cdf_=cdf_, xmin=xmin, xmax=xmax)
    F_xmin, F_xmax = cdf(np.array([xmin, xmax]))
    ppf_: Callable = scipy.interpolate.interp1d(empirical_cdf, empirical_range, interpolation_method, bounds_error=False)
    ppf: Callable = partial(numerical_ppf, ppf_=ppf_, xmin=xmin, xmax=xmax, F_xmin=F_xmin, F_xmax=F_xmax)
    support: Callable = partial(numerical_support, xmin=xmin, xmax=xmax)

    return pdf, cdf, ppf, support, None


def calc_num_bins(num_unique_points: int) -> int:
    """Function used to determine the number of bins to use when producing the empirical pdf values via a histogram.
    Works on the idea that we should have more bins if we have more data.
    """
    if num_unique_points >= 10000:
        num_bins: int = 100
    elif num_unique_points > 2000:
        # if spread evenly, roughly at least 100 unique data points in each bin
        num_bins: int = math.floor(0.01 * (num_unique_points))
    elif num_unique_points >= 100:
        # if spread evenly, roughly at least 10 unique data points in each bin
        num_bins: int = math.floor((180 / 19) + (num_unique_points / 190))
    else:
        num_bins: int = min(num_unique_points, 10)
    return num_bins


def continuous_empirical_fit(data: np.ndarray) -> tuple:
    """Fitting function for a univariate continuous empirical distribution.

    Parameters
    ===========
    data : np.ndarray
        The data to fit to the continuous empirical distribution in a flattened numpy array.

    Returns
    =======
    fitted_funcs: tuple
        fitted pdf, cdf, ppf, support, rvs functions
    """
    interpolate_method: str = 'linear'  # 'cubic' # cubic leads to overfitting
    N: int = len(data)

    # determining the number of bins to use to calculate pdf values.
    num_unique_points: int = len(set(data))
    num_bins: int = calc_num_bins(num_unique_points)

    xmin, xmax = data.min(), data.max()
    bin_width: float = (xmax - xmin) / num_bins

    # Calculating empirical pdf and cdf values
    empirical_pdf, empirical_range = np.histogram(data, bins=num_bins)
    empirical_range = (empirical_range[1:] + empirical_range[:-1]) / 2
    empirical_range[0] = xmin
    empirical_range[-1] = xmax
    empirical_pdf = empirical_pdf / (N * bin_width)
    empirical_cdf: np.ndarray = np.cumsum(empirical_pdf)
    empirical_cdf = empirical_cdf / empirical_cdf[-1]

    # Generating pdf, cdf and ppf functions via interpolation
    pdf_: Callable = scipy.interpolate.interp1d(empirical_range, empirical_pdf, interpolate_method, bounds_error=False)
    pdf: Callable = partial(numerical_pdf, pdf_=pdf_)

    _, idx = np.unique(empirical_cdf, return_index=True)  # removing duplicates (we may have multiple 0's and 1's)
    empirical_range, empirical_cdf = empirical_range[idx], empirical_cdf[idx]

    # fitting cdf, ppf and support functions
    cdf_: Callable = scipy.interpolate.interp1d(empirical_range, empirical_cdf, interpolate_method, bounds_error=False)
    cdf: Callable = partial(numerical_cdf, cdf_=cdf_, xmin=empirical_range[0], xmax=empirical_range[-1])
    F_xmin, F_xmax = cdf(np.array([xmin, xmax]))
    ppf_: Callable = scipy.interpolate.interp1d(empirical_cdf, empirical_range, interpolate_method, bounds_error=False)
    ppf: Callable = partial(numerical_ppf, ppf_=ppf_, xmin=xmin, xmax=xmax, F_xmin=F_xmin, F_xmax=F_xmax)
    support: Callable = partial(numerical_support, xmin=xmin, xmax=xmax)

    return pdf, cdf, ppf, support, None

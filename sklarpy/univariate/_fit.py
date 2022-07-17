# Contains fitting functions for univariate distributions.
import numpy as np
import scipy.interpolate
import scipy.stats
import math


__all__ = ['dlaplace_fit', 'duniform_fit', 'geometric_fit', 'planck_fit', 'poisson_fit', 'kde_fit', 'discrete_empirical_fit', 'continuous_empirical_fit']


########################################################################################################################
# Parametric
########################################################################################################################


def dlaplace_fit(x: np.ndarray) -> tuple:
    """Fitting function for the discrete laplacian distribution. Returns the MLE estimator of the parameter alpha.

    Math
    -----
    alpha_mle = arsinh( n/sum(|x_i|) )

    See Also:
    ---------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dlaplace.html#scipy.stats.dlaplace

    Returns
    =======
    mle_estimator: tuple
       (alpha, )
    """
    return math.asinh(len(x)/abs(x).sum()),


def duniform_fit(x: np.ndarray) -> tuple:
    """Fitting function for the discrete uniform distribution. Returns the minimum and maximum values of the sample.

    Math
    -----
    a = min(x)
    b = max(x)

    See Also:
    ---------
    https://en.wikipedia.org/wiki/Discrete_uniform_distribution

    Returns
    =======
    estimator: tuple
       (a, b)
    """
    return x.min(), x.max() + 1


def geometric_fit(x: np.ndarray) -> tuple:
    """Fitting function for the geometric distribution. Returns the MLE estimator of the parameter p.

    Math
    -----
    p_mle = n / sum(x)

    See Also:
    ---------
    https://en.wikipedia.org/wiki/Geometric_distribution

    Returns
    =======
    mle_estimator: tuple
       (p, )
    """
    return len(x) / sum(x),


def planck_fit(x: np.ndarray) -> tuple:
    """Fitting function for the planck exponential distribution. Returns the MLE estimator of the parameter lambda.

    Math
    -----
    lambda_mle = ln(1 + n/sum(x))

    See Also:
    ---------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.planck.html#scipy.stats.planck

    Returns
    =======
    mle_estimator: tuple
       (lambda, )
    """
    return np.log(1+(len(x) / x.sum())),


def poisson_fit(x: np.ndarray):
    """Fitting function for the poisson distribution. Returns the MLE estimator of the parameter lambda.

    Math
    -----
    lambda_mle = sum(x) / n

    See Also:
    ---------
    https://en.wikipedia.org/wiki/Poisson_distribution

    Returns
    =======
    mle_estimator: tuple
       (lambda, )
    """
    return x.mean(),


########################################################################################################################
# Numerical/Non-Parametric
########################################################################################################################

def kde_fit(data: np.ndarray):
    """Fitting function for a univariate gaussian kernel density estimator distribution.

    Returns
    =======
    fitted_funcs: tuple
        pdf, cdf, ppf, support, rvs
    """
    kde = scipy.stats.gaussian_kde(data)
    pdf = kde.pdf
    cdf = (lambda x: np.array([kde.integrate_box_1d(-np.inf, v) for v in x]))
    empirical_range = np.linspace(data.min(), data.max(), 100)
    ppf = scipy.interpolate.interp1d(cdf(empirical_range), empirical_range, 'cubic', bounds_error=False)
    support: tuple = (-np.inf, np.inf)
    # rvs = (lambda size: kde.resample(size))
    return pdf, cdf, ppf, support, kde.resample


def discrete_empirical_fit(data: np.ndarray):
    """Fitting function for a univariate discrete empirical distribution.

    Returns
    =======
    fitted_funcs: tuple
        pdf, cdf, ppf, support, rvs
    """
    support = data.min(), data.max()
    N: int = len(data)

    pdf = (lambda x: np.array([np.count_nonzero(data == i) / N for i in x]))

    # Generating cdf and ppf functions via interpolation
    empirical_range: np.ndarray = np.arange(support[0], support[1] + 1)
    empirical_cdf = np.cumsum(pdf(empirical_range))
    _, idx = np.unique(empirical_cdf, return_index=True)
    empirical_range, empirical_cdf = empirical_range[idx], empirical_cdf[idx]
    cdf = scipy.interpolate.interp1d(empirical_range, empirical_cdf, 'nearest', bounds_error=False)

    ppf = scipy.interpolate.interp1d(empirical_cdf, empirical_range, 'nearest', bounds_error=False)

    return pdf, cdf, ppf, support, None


def continuous_empirical_fit(data: np.ndarray) -> tuple:
    """Fitting function for a univariate continuous empirical distribution.

    Returns
    =======
    fitted_funcs: tuple
        pdf, cdf, ppf, support, rvs
    """
    N: int = len(data)
    data_std = np.std(data)/10

    num_bins: int = min(len(set(data)), 20)  # set to 20 to allow for some smoothing
    bin_width = (data.max() - data.min()) / num_bins
    empirical_pdf, empirical_range = np.histogram(data, bins=num_bins)
    empirical_range = (empirical_range[1:] + empirical_range[:-1]) / 2
    empirical_cdf = np.cumsum(empirical_pdf) / N
    empirical_pdf = empirical_pdf / (N * bin_width)

    # making sure empirical dist tails off to 0
    empirical_range = np.append(empirical_range[0] - data_std, empirical_range)
    empirical_pdf = np.append(0, empirical_pdf)
    empirical_cdf = np.append(0, empirical_cdf)
    empirical_range = np.append(empirical_range, empirical_range[-1] + data_std)
    empirical_pdf = np.append(empirical_pdf, 0)
    empirical_cdf = np.append(empirical_cdf, 1)

    # Generating pdf, cdf and ppf functions via interpolation
    pdf_ = scipy.interpolate.interp1d(empirical_range, empirical_pdf, 'cubic', bounds_error=False)
    pdf = (lambda x: np.clip(pdf_(x), 0.0, 1.0))

    _, idx = np.unique(empirical_cdf, return_index=True)
    empirical_range, empirical_cdf = empirical_range[idx], empirical_cdf[idx]
    cdf_ = scipy.interpolate.interp1d(empirical_range, empirical_cdf, 'cubic', bounds_error=False)
    cdf = (lambda x: np.clip(cdf_(x), 0.0, 1.0))

    ppf = scipy.interpolate.interp1d(empirical_cdf, empirical_range, 'cubic', bounds_error=False)
    return pdf, cdf, ppf, (empirical_range[0], empirical_range[-1]), None

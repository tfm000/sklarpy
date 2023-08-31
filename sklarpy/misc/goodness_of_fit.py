# Contains goodness-of-fit tests
from typing import Callable
from functools import partial
from scipy.stats import cramervonmises, kstest, chi2
import pandas as pd
import numpy as np

__all__ = ['continuous_gof', 'discrete_gof']


def continuous_gof(data: np.ndarray, params: tuple, cdf: Callable,
                   name: str = 'gof') -> pd.DataFrame:
    """Compute goodness of fit tests for a continuous distribution.

    Parameters
    ----------
    data : np.ndarray
        The data being fitted by the continuous distribution.
    params : tuple
        The parameters used to fit the data.
    cdf : Callable
        The cdf function of the continuous distribution. Must take a numpy
        array containing data and the distribution's parameters in a tuple as
        arguments.
    name: str
        The name of the distribution. If non-provided, 'gof' will be used.

    Returns
    -------
    gof: pd.DataFrame
        A dataframe containing the goodness of fit tests for the continuous
        distribution.

    Notes
    -----
    Cramér-von Mises gof test
    https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93von_Mises_criterion

    Kolmogorov-Smirnov gof test
    https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    """
    cdf = partial(cdf, params=params)

    # Cramér-von Mises gof test
    cvm_res = cramervonmises(data, cdf)

    # Checking if we reject H0 at various levels of confidence
    cvm_significant_10pct: bool = cvm_res.pvalue >= 0.1
    cvm_significant_5pct: bool = cvm_res.pvalue >= 0.05
    cvm_significant_1pct: bool = cvm_res.pvalue >= 0.01

    # Kolmogorov-Smirnov gof test
    ks_stat, ks_pvalue = kstest(data, cdf)

    # Checking if we reject H0 at various levels of confidence
    ks_significant_10pct: bool = ks_pvalue >= 0.1
    ks_significant_5pct: bool = ks_pvalue >= 0.05
    ks_significant_1pct: bool = ks_pvalue >= 0.01

    # Creating dataframe containing goodness of fit results
    values = [cvm_res.statistic, cvm_res.pvalue, cvm_significant_10pct,
              cvm_significant_5pct, cvm_significant_1pct, ks_stat, ks_pvalue,
              ks_significant_10pct, ks_significant_5pct, ks_significant_1pct]
    index = ['Cramér-von Mises statistic', 'Cramér-von Mises p-value',
             'Cramér-von Mises @ 10%', 'Cramér-von Mises @ 5%',
             'Cramér-von Mises @ 1%', 'Kolmogorov-Smirnov statistic',
             'Kolmogorov-Smirnov p-value', 'Kolmogorov-Smirnov @ 10%',
             'Kolmogorov-Smirnov @ 5%', 'Kolmogorov-Smirnov @ 1%']
    return pd.DataFrame(values, index=index, columns=[name])


def discrete_gof(data: np.ndarray, params: tuple,
                 support: Callable, pdf: Callable, ppf: Callable,
                 name: str = 'gof') -> pd.DataFrame:
    """
    Compute goodness of fit tests for a discrete distribution.

    Parameters
    ----------
    data : np.ndarray
        The data being fitted by the discrete distribution.
    params : tuple
        The parameters used to fit the data.
    support : Callable
        A function which generates/calculates the support of the discrete
        distribution. Must take the distributions parameters in a tuple as an
        argument.
    pdf: Callable
        The pdf function of the discrete distribution. Must take a numpy array
        containing data and the distribution's
        parameters in a tuple as arguments.
    ppf: Callable
        The inverse function of the discrete distribution. Must take a numpy
        array containing quantile values and the
        distribution's parameters in a tuple as arguments.
    name: str
        The name of the distribution. If non-provided, 'gof' will be used.

    Returns
    -------
    gof: pd.DataFrame
        A dataframe containing the goodness of fit tests for the discrete
        distribution.

    Notes
    -----
    Chi-squared gof test

    """
    num: int = len(data)
    dof: int = num - len(params) - 1

    xmin, xmax = support(params)
    eps: float = 10 ** -4
    xmin: int = int(max(xmin, ppf(eps, params)))
    xmax: int = int(min(xmax, ppf(1 - eps, params)))

    # Chi-squared gof test
    xrange = np.arange(xmin, xmax + 1, dtype=int)
    observed = np.array([np.count_nonzero(data == x) for x in xrange])
    expected = pdf(xrange, params) * num
    index = np.where(expected != 0)[0]
    expected = expected[index]
    observed = observed[index]
    chisq_stat = np.sum(((expected - observed) ** 2) / expected)
    chisq_pvalue = chi2.sf(chisq_stat, dof)

    # Checking if we reject H0 at various levels of confidence
    chisq_significant_10pct: bool = chisq_pvalue >= 0.1
    chisq_significant_5pct: bool = chisq_pvalue >= 0.05
    chisq_significant_1pct: bool = chisq_pvalue >= 0.01

    # Creating dataframe containing goodness of fit results
    values = [float(chisq_stat), float(chisq_pvalue), chisq_significant_10pct,
              chisq_significant_5pct, chisq_significant_1pct]
    index = ['chi-square statistic', 'chi-square p-value', 'chi-square @ 10%',
             'chi-square @ 5%', 'chi-square @ 1%']
    return pd.DataFrame(values, index=index, columns=[name])

from typing import Callable
from functools import partial

from scipy.stats import cramervonmises, kstest, chi2
import pandas as pd
import numpy as np

from sklarpy._utils import check_params, check_univariate_data, prob_bounds


def continuous_goodness_of_fit(data: np.ndarray, params: tuple, cdf: Callable, name: str) -> pd.DataFrame:
    cdf = partial(cdf, params=params)

    # Cramér-von Mises gof test
    cvm_res = cramervonmises(data, cdf)

    # Kolmogorov-Smirnov gof test
    ks_stat, ks_pvalue = kstest(data, cdf)

    # Creating dataframe containing goodness of fit results
    values = [cvm_res.statistic, cvm_res.pvalue, ks_stat, ks_pvalue]
    index = ['Cramér-von Mises statistic', 'Cramér-von Mises p-value',
             'Kolmogorov-Smirnov statistic', 'Kolmogorov-Smirnov p-value']
    return pd.DataFrame(values, index=index, columns=[name])


def discrete_goodness_of_fit(data: np.ndarray, params: tuple, support: Callable, pdf: Callable, ppf: Callable, name: str) -> pd.DataFrame:
    num: int = len(data)
    dof: int = num - len(params) - 1

    xmin, xmax = support(params)
    xmin: int = int(max(xmin, ppf(prob_bounds[0], params)))
    xmax: int = int(min(xmax, ppf(prob_bounds[1], params)))

    # Chi-squared gof test
    xrange = np.arange(xmin, xmax + 1, dtype=int)
    observed = np.array([np.count_nonzero(data == x) for x in xrange])
    expected = pdf(xrange, params) * num
    index = np.where(expected != 0)[0]
    expected = expected[index]
    observed = observed[index]
    chisq_stat = np.sum(((expected - observed) ** 2) / expected)
    chisq_pvalue = chi2.sf(chisq_stat, dof)
    values = [float(chisq_stat), float(chisq_pvalue)]
    index = ['chi-square statistic', 'chi-square p-value']
    return pd.DataFrame(values, index=index, columns=[name])

import numpy as np
from collections import deque
from scipy.integrate import nquad
import scipy.stats

from sklarpy._utils import get_iterator

__all__ = ['multivariate_t_cdf']


def multivariate_t_cdf(x: np.ndarray, loc=None, shape=1, df=1, allow_singular: bool = False, show_progress: bool = True) -> np.ndarray:
    num_rows, num_variables = x.shape

    def multivariate_t_integratable_pdf(*xrow):
        return scipy.stats.multivariate_t.pdf(xrow, loc, shape, df, allow_singular)

    def singlular_cdf(xrow: np.ndarray) -> float:
        ranges = [[-np.inf, float(xrow[i])] for i in range(num_variables)]
        res: tuple = nquad(multivariate_t_integratable_pdf, ranges)
        return res[0]

    cdf_values: deque = deque()
    iterator = get_iterator(x, show_progress, "calculating cdf values")
    for xrow in iterator:
        val: float = singlular_cdf(xrow)
        cdf_values.append(val)
    return np.asarray(cdf_values, dtype=float)

import scipy.stats
import numpy as np
from functools import partial
from collections import deque

from sklarpy.copulas.numerical._numerical import Numerical
from sklarpy.copulas.numerical._numerical_params import GaussianKdeCopulaParams

__all__ = ['GaussianKdeCopula']


class GaussianKdeCopula(Numerical):
    _OBJ_NAME = "GaussianKdeCopula"
    _MAX_NUM_VARIABLES = np.inf
    _PARAMS_OBJ = GaussianKdeCopulaParams

    def __cdf_func(self, u: np.ndarray, kde: scipy.stats.gaussian_kde) -> np.ndarray:
        ninfs: np.ndarray = np.full((self.num_variables, ), -np.inf)
        cdf_vals: deque = deque()
        for row in u:
            cdf_vals.append(kde.integrate_box(ninfs, row))
        return np.asarray(cdf_vals)

    def fit(self, params: GaussianKdeCopulaParams = None, **kwargs):
        # kwargs to pass to gaussian_kde
        self._fitting = True

        if not self._fit_params(params):
            # User fif not provide params

            if self._marginals is None:
                raise ValueError("if params are not specified, marginal cdf values must be provided in init.")

            shape: tuple = self._marginals.shape
            self._check_num_variables(shape[1])

            # generating additional data points
            self._umins, self._umaxs = self._marginals.min(axis=0), self._marginals.max(axis=0)

            # fitting
            kde: scipy.stats.gaussian_kde = scipy.stats.gaussian_kde(self._marginals.T)
            self._pdf_func = kde.pdf
            self._cdf_func = partial(self.__cdf_func, kde=kde)

        self._params = {'pdf_func': self._pdf_func, 'cdf_func': self._cdf_func, 'umins': self._umins,
                        'umaxs': self._umaxs, 'num_variables': self._num_variables}
        self._fitting = False
        self._fitted = True

        return self


if __name__ == '__main__':
    from sklarpy import load
    std6 = load('std6')
    rvs = std6.rvs(5000)
    gkde = GaussianKdeCopula(rvs)
    gkde.fit()
    std6.pdf_plot()
    # gkde.pdf_plot()
    # gkde.cdf_plot()
    # gkde.marginal_pairplot(alpha=0.1)
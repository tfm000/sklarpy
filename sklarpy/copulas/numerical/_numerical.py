from typing import Callable
import numpy as np

from sklarpy.copulas._copula import Copula

__all__ = ['Numerical']


class Numerical(Copula):
    def __init__(self, marginals=None, name: str = None):
        Copula.__init__(self, marginals, name)
        self._pdf_func: Callable = None
        self._cdf_func: Callable = None
        self._umins: np.ndarray = None
        self._umaxs: np.ndarray = None

    def _fit_params(self, params) -> bool:
        if params is not None:
            self._params_check(params)
            for param in ('pdf_func', 'cdf_func', 'umins', 'umaxs', 'num_variables'):
                if param not in params:
                    raise ValueError(f"{params} is not params")
            self._pdf_func = params.pdf_func
            self._cdf_func = params.cdf_func
            self._umins = params.umins
            self._umaxs = params.umaxs
            self._num_variables = params
            return True
        return False

    def _pdf(self, u: np.ndarray, **kwargs) -> np.ndarray:
        if u.ndim == 1:
            u = u.reshape((1, u.size))
        raw_vals: np.ndarray = self._pdf_func(u)
        pdf_vals: np.ndarray = np.where(~np.isnan(raw_vals), raw_vals, 0.0)
        return np.clip(pdf_vals, 0.0, None)

    def _cdf(self, u: np.ndarray, **kwargs) -> np.ndarray:
        if u.ndim == 1:
            u = u.reshape((1, u.size))

        for var in range(self._num_variables):
            u[:, var] = np.where(u[:, var] >= self._umins[var], u[:, var], self._umins[var])
            u[:, var] = np.where(u[:, var] <= self._umaxs[var], u[:, var], self._umaxs[var])
        return self._cdf_func(u)

    def _rvs(self, size: int) -> np.ndarray:
        """Using a bootstrapping approach"""
        if self._marginals is None:
            raise NotImplementedError(f"Cannot implement rvs for {self._OBJ_NAME} if self._marginals is specified.")
        index: np.ndarray = np.random.randint(0, self._marginals.shape[0], size=size)
        return self._marginals[index, :]

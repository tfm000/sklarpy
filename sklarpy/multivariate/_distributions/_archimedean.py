# contains code for archimedean copulas
import numpy as np
from abc import abstractmethod
from typing import Union, Tuple
import pandas as pd
import scipy.stats

from sklarpy.multivariate._prefit_dists import PreFitContinuousMultivariate, FittedContinuousMultivariate
from sklarpy._utils import get_iterator, FitError, check_multivariate_data
from sklarpy._other import Params

# params_example = ('theta', 'dim') -> dim only needed for non-bivariate copulas
__all__ = ['multivariate_clayton_gen']


class multivariate_archimedean_base_gen(PreFitContinuousMultivariate):
    _DATA_FIT_METHODS = ("mle", 'inverse_kendall_tau', 'low_dim_mle')
    _DEFAULT_STRICT_BOUNDS: tuple
    _DEFAULT_BOUNDS: tuple

    @abstractmethod
    def _generator(self, u: np.ndarray, params: tuple) -> np.ndarray:
        pass

    @abstractmethod
    def _generator_inverse(self, t: np.ndarray, params: tuple) -> np.ndarray:
        pass

    def _cdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        shape: tuple = x.shape

        show_progress: bool = kwargs.get('show_progress', True)
        iterator = get_iterator(range(shape[1]), show_progress, "calculating cdf values")

        t: np.ndarray = np.zeros((shape[0], ), dtype=float)
        for i in iterator:
            t += self._generator(u=x[:, i], params=params)
        return self._generator_inverse(t=t, params=params)

    def _G_hat(self, t: np.ndarray, params: tuple) -> np.ndarray:
        return self._generator_inverse(t=t, params=params)

    @abstractmethod
    def _v_rvs(self, size: int, params: tuple) -> np.ndarray:
        pass

    def _get_dim(self, params: tuple) -> int:
        return 2

    def _rvs(self, size: int, params: tuple) -> np.ndarray:
        v: np.ndarray = self._v_rvs(size=size, params=params)
        d: int = self._get_dim(params=params)
        x: np.ndarray = np.random.uniform(size=(size, d))
        t: np.ndarray = -np.log(x) / v
        return self._G_hat(t=t, params=params)

    def _get_bounds(self, data: np.ndarray, as_tuple: bool, **kwargs) -> Union[dict, tuple]:
        d: int = data.shape[1]
        theta_bounds: tuple = self._DEFAULT_STRICT_BOUNDS if d > 2 else self._DEFAULT_BOUNDS
        default_bounds: dict = {'theta': theta_bounds}
        return super()._get_bounds(default_bounds, d, as_tuple, **kwargs)

    def _get_low_dim_theta0(self, data: np.ndarray, bounds: tuple, copula: bool) -> np.ndarray:
        theta0: float = np.random.uniform(*bounds[0])
        return np.array([theta0], dtype=float)

    def _low_dim_theta_to_params(self, theta: np.ndarray, d: int) -> tuple:
        return theta[0], d

    def _get_low_dim_mle_objective_func_args(self, data: np.ndarray, **kwargs) -> tuple:
        return data.shape[1],

    @abstractmethod
    def _inverse_kendall_tau(self, data: np.ndarray, **kwargs) -> float:
        d: int = data.shape[1]
        if d != 2:
            raise FitError("Archimedean copulas can only be fit using inverse kendall's tau when the number of variables is exactly 2.")

        return scipy.stats.kendalltau(data[:, 0], data[:, 1]).statistic

    def _mle(self, data: np.ndarray, **kwargs) -> tuple:
        return super()._low_dim_mle(data, **kwargs)

    def _fit_given_data_kwargs(self, method: str, data: np.ndarray, **user_kwargs) -> dict:
        if method in ('mle', 'low_dim_mle'):
            return super()._fit_given_data_kwargs('low_dim_mle', data, **user_kwargs)
        return {'copula': True}

    def _fit_given_params_tuple(self, params: tuple, **kwargs) -> Tuple[dict, int]:
        return {'theta': params[0]}

    def fit(self, data: Union[np.ndarray, pd.DataFrame] = None, params: Union[Params, tuple] = None, method: str = None, **kwargs) -> FittedContinuousMultivariate:
        if method is None and data is not None:
            data_array: np.ndarray = check_multivariate_data(data, allow_1d=True)
            d: int = data_array.shape[1]
            method = 'inverse_kendall_tau' if d == 2 else 'mle'
        return super().fit(data=data, params=params, method=method, **kwargs)

    def num_scalar_params(self, d: int, copula: bool, **kwargs) -> int:
        return 2


class multivariate_clayton_gen(multivariate_archimedean_base_gen):
    _DEFAULT_STRICT_BOUNDS = (0, 100.0)
    _DEFAULT_BOUNDS = (-1, 100.0)

    def _generator(self, u: np.ndarray, params: tuple) -> np.ndarray:
        theta: float = params[0]
        return (np.power(u, -theta) - 1) / theta

    def _generator_inverse(self, t: np.ndarray, params: tuple) -> np.ndarray:
        theta: float = params[0]
        return np.power((theta * t) + 1, -1 / theta)

    def _G_hat(self, t: np.ndarray, params: tuple) -> np.ndarray:
        theta: float = params[0]
        return np.power(t + 1, -1 / theta)

    def _v_rvs(self, size: int, params: tuple) -> np.ndarray:
        theta = params[0]
        if theta < 0:
            raise NotImplementedError("cannot generate random variables for Clayton Copula when theta parameter is not positive.")
        return scipy.stats.gamma.rvs(a=1/theta, scale=1, size=(size, 1))

    def _get_dim(self, params: tuple) -> int:
        return params[1]

    def _inverse_kendall_tau(self, data: np.ndarray, **kwargs) -> tuple:
        kendall_tau: float = super()._inverse_kendall_tau(data=data, **kwargs)
        theta: float = 2 * kendall_tau / (1 - kendall_tau)
        d: int = data.shape[1]
        return (theta, d), ~np.isinf(theta)

    def _fit_given_params_tuple(self, params: tuple, **kwargs) -> Tuple[dict, int]:
        return {'theta': params[0], 'd': params[1]}, params[1]

    def _logpdf(self, x: np.ndarray, params: tuple,  **kwargs) -> np.ndarray:
        theta, d = params

        # common calculations
        theta_inv: float = 1/theta

        # calculating evaluating generator function
        gen_sum: np.ndarray = np.zeros((x.shape[0], ), dtype=float)
        log_cs: float = 0.0
        log_gs: float = 0.0
        for i in range(d):
            gen_val: np.ndarray = self._generator(u=x[:, i], params=params)
            gen_sum += gen_val

            log_cs += np.log(theta_inv + d - i)
            log_gs += np.log((theta*gen_val) + 1)

        return (d * np.log(theta)) - ((theta_inv + d) * np.log((theta * gen_sum) + 1)) + log_cs + ((theta_inv + 1) * log_gs)


class multivariate_gumbel_gen(multivariate_archimedean_base_gen):
    _DEFAULT_STRICT_BOUNDS = (1.01, 100.0)
    _DEFAULT_BOUNDS = (1.01, 100.0)

    def _generator(self, u: np.ndarray, params: tuple) -> np.ndarray:
        theta: float = params[0]
        return np.power(-np.log(u), theta)

    def _generator_inverse(self, t: np.ndarray, params: tuple) -> np.ndarray:
        theta: float = params[0]
        return np.exp(-np.power(t, 1/theta))

    def _v_rvs(self, size: int, params: tuple) -> np.ndarray:
        theta: float = params[0]
        c: float = np.power(np.cos(np.pi/(2*theta)), theta)
        return scipy.stats.levy_stable.rvs(1/theta, 1.0, c, 0.0, size=(size, 1))

    def _get_dim(self, params: tuple) -> int:
        return params[1]

    def _inverse_kendall_tau(self, data: np.ndarray, **kwargs) -> tuple:
        kendall_tau: float = super()._inverse_kendall_tau(data=data, **kwargs)
        theta: float = 1 / (1 - kendall_tau)
        d: int = data.shape[1]
        return (theta, d), ~np.isinf(theta)

    def _fit_given_params_tuple(self, params: tuple, **kwargs) -> Tuple[dict, int]:
        return {'theta': params[0], 'd': params[1]}, params[1]

    def _DK_psi(self, t: np.ndarray, params: tuple, K: int) -> np.ndarray:
        if K == 0:
            return self._generator_inverse(t=t, params=params)
        theta: float = params[0]
        theta_inv: float = 1/theta

        DK_psi: np.ndarray = np.zeros(t.shape, dtype=float)
        for j in range(K):
            prod = np.array([theta_inv - i + 1 for i in range(1, K - j + 1)]).prod()
            DK_psi -= self._DK_psi(t=t, params=params, K=j) * np.power(t, theta_inv - K + j) * prod
        return DK_psi

    def _logpdf(self, x: np.ndarray, params: tuple,  **kwargs) -> np.ndarray:
        theta, d = params

        # calculating evaluating generator function
        gen_sum: np.ndarray = np.zeros((x.shape[0],), dtype=float)
        log_gs: float = 0.0
        for i in range(d):
            gen_val: np.ndarray = self._generator(u=x[:, i], params=params)
            gen_sum += gen_val

            log_gs -= (((1/theta) - 1) * np.log(gen_val)) + np.log(x[:, i])

        # calculating d-th derivative of generator inverse
        Dd_psi: np.ndarray = self._DK_psi(t=gen_sum, params=params, K=d)
        return np.log(np.abs(Dd_psi)) + (d*np.log(theta)) + log_gs

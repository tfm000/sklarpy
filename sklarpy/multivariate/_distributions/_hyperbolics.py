# Contains code for multivariate Hyperbolic models
import numpy as np
from typing import Tuple, Union
from scipy.optimize import differential_evolution
from abc import abstractmethod

from sklarpy.multivariate._distributions._generalized_hyperbolic import \
    multivariate_gen_hyperbolic_gen
from sklarpy.multivariate._prefit_dists import PreFitContinuousMultivariate
from sklarpy.utils._params import Params

__all__ = ['multivariate_marginal_hyperbolic_gen',
           'multivariate_hyperbolic_gen', 'multivariate_nig_gen']


class multivariate_hyperbolic_base_gen(multivariate_gen_hyperbolic_gen):
    """Base class for multivariate Hyperbolic models."""
    _NUM_W_PARAMS: int = 2

    @abstractmethod
    def _init_lamb(self, *args) -> None:
        """Initialize the _lamb attribute with the equivalent Generalized
        Hyperbolic lambda parameter for the model."""

    def _check_params(self, params: tuple, **kwargs) -> None:
        # adjusting params to fit multivariate generalized hyperbolic params
        num_params: int = len(params)
        if num_params == 5:
            params = self._get_params(params, check_params=False)
        elif num_params != 6:
            raise ValueError("Incorrect number of params given by user")
        self._num_params = 6

        # checking params
        super()._check_params(params)
        self._num_params = 5

    def _get_params(self, params: Union[Params, tuple], **kwargs) -> tuple:
        params_tuple: tuple = PreFitContinuousMultivariate._get_params(
            self, params, **kwargs)
        self._init_lamb(params_tuple[-1].shape[0])
        params_tuple = self._lamb, *params_tuple
        return params_tuple[-6:]

    def _get_bounds(self, data: np.ndarray, as_tuple: bool = True, **kwargs
                    ) -> Union[dict, tuple]:
        bounds_dict: dict = super()._get_bounds(data, False, **kwargs)
        # removing lambda from bounds
        return self._remove_bounds(bounds_dict, ['lamb'],
                                   data.shape[1], as_tuple)

    def _add_randomness(self, params: tuple, bounds: tuple, d: int,
                        randomness_var: float, copula: bool) -> tuple:
        adj_params: tuple = super()._add_randomness(
            params=params, bounds=bounds, d=d,
            randomness_var=randomness_var, copula=copula)
        return self._get_params(adj_params, check_params=False)

    def _neg_q2(self, w_params: np.ndarray, etas: np.ndarray,
                deltas: np.ndarray, zetas: np.ndarray) -> float:
        return super()._neg_q2((self._lamb, *w_params), etas, deltas, zetas)

    def _q2_opt(self, bounds: tuple, etas: np.ndarray, deltas: np.ndarray,
                zetas: np.ndarray, q2_options: dict):
        q2_res = differential_evolution(
            self._neg_q2, bounds=bounds[:2],
            args=(etas, deltas, zetas), **q2_options)
        chi, psi = q2_res['x']
        return {'x': np.array([self._lamb, chi, psi]),
                'success': q2_res['success']}

    def _gh_to_params(self, params: tuple) -> tuple:
        return params[1:]

    def _theta_to_params(self, theta: np.ndarray, mean: np.ndarray,
                         S: np.ndarray, S_det: float, min_eig: float,
                         copula: bool, **kwargs) -> tuple:
        # modifying theta to fit that of the Generalized Hyperbolic
        theta = np.array([self._lamb, *theta], dtype=float)
        return super()._theta_to_params(theta=theta, mean=mean, S=S,
                                        S_det=S_det, min_eig=min_eig,
                                        copula=copula, **kwargs)

    def _params_to_theta(self, params: tuple, **kwargs) -> np.ndarray:
        return np.array([*params[1:3], *params[-1].flatten()], dtype=float)

    def _get_params0(self, data: np.ndarray, bounds: tuple, cov_method: str,
                     min_eig, copula: bool, **kwargs) -> tuple:
        # modifying bounds to fit those of the Generalized Hyperbolic
        d: int = (len(bounds) - 2) / 2 if self._ASYMMETRIC else len(bounds) - 2
        d = int(d)
        self._init_lamb(d)
        bounds = ((self._lamb, self._lamb), *bounds)

        return super()._get_params0(data=data, bounds=bounds,
                                    cov_method=cov_method, min_eig=min_eig,
                                    copula=copula, **kwargs)

    def _fit_given_params_tuple(self, params: tuple, **kwargs
                                ) -> Tuple[dict, int]:
        self._check_params(params, **kwargs)
        return {'chi': params[0], 'psi': params[1], 'loc': params[2],
                'shape': params[3], 'gamma': params[4]}, params[2].size


class multivariate_marginal_hyperbolic_gen(multivariate_hyperbolic_base_gen):
    """Multivariate Marginal Hyperbolic model."""
    def _init_lamb(self, *args):
        self._lamb: float = 1.0


class multivariate_hyperbolic_gen(multivariate_hyperbolic_base_gen):
    """Multivariate Hyperbolic model."""
    def _init_lamb(self, d: int):
        self._lamb: float = 0.5 * (d + 1)


class multivariate_nig_gen(multivariate_hyperbolic_base_gen):
    """Multivariate Normal-Inverse Gaussian (NIG) model."""
    def _init_lamb(self, *args):
        self._lamb: float = -0.5

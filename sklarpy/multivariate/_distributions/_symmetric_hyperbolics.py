# Contains code for multivariate symmetric Hyperbolic models
import numpy as np
from typing import Tuple, Union

from sklarpy.multivariate._distributions._hyperbolics import \
    multivariate_hyperbolic_base_gen
from sklarpy.multivariate._prefit_dists import PreFitContinuousMultivariate
from sklarpy.utils._params import Params

__all__ = ['multivariate_sym_marginal_hyperbolic_gen',
           'multivariate_sym_hyperbolic_base_gen', 'multivariate_sym_nig_gen']


class multivariate_sym_hyperbolic_base_gen(multivariate_hyperbolic_base_gen):
    """Multivariate Symmetric Hyperbolic Base class."""
    _ASYMMETRIC = False

    def _check_params(self, params: tuple, **kwargs) -> None:
        # adjusting params to fit multivariate generalized hyperbolic params
        num_params: int = len(params)
        if num_params == 4:
            params = self._get_params(params, check_params=False)
        elif num_params != 6:
            raise ValueError("Incorrect number of params given by user")
        self._num_params = num_params

        # checking params
        super()._check_params(params)
        self._num_params = 4

    def _get_params(self, params: Union[Params, tuple], **kwargs) -> tuple:
        params_tuple: tuple = PreFitContinuousMultivariate._get_params(
            self, params, **kwargs)
        self._init_lamb(params_tuple[-1].shape[0])
        if len(params_tuple) == 6:
            return params_tuple

        # 4 params
        loc: np.ndarray = params_tuple[2]
        gamma: np.ndarray = np.zeros(loc.shape, dtype=float)
        return self._lamb, *params_tuple, gamma

    def _get_bounds(self, data: np.ndarray, as_tuple: bool = True, **kwargs) \
            -> Union[dict, tuple]:
        bounds_dict: dict = super()._get_bounds(data, False, **kwargs)
        # removing gamma from bounds
        return self._remove_bounds(bounds_dict, ['gamma'], data.shape[1],
                                   as_tuple)

    def _gh_to_params(self, params: tuple) -> tuple:
        return params[1:5]

    def _theta_to_params(self, theta: np.ndarray, mean: np.ndarray,
                         S: np.ndarray, S_det: float, min_eig: float,
                         copula: bool, **kwargs) -> tuple:

        # modifying theta to fit that of the Generalized Hyperbolic
        d: int = mean.size
        theta: np.ndarray = np.array([*theta, *np.zeros((d,))], dtype=float)
        return super()._theta_to_params(theta=theta, mean=mean, S=S,
                                        S_det=S_det, min_eig=min_eig,
                                        copula=copula, **kwargs)

    def _params_to_theta(self, params: tuple, **kwargs) -> np.ndarray:
        return np.array([*params[1:3]], dtype=float)

    def _fit_given_params_tuple(self, params: tuple, **kwargs) \
            -> Tuple[dict, int]:
        self._check_params(params, **kwargs)
        return {'chi': params[0], 'psi': params[1], 'loc': params[2],
                'shape': params[3]}, params[2].size


class multivariate_sym_marginal_hyperbolic_gen(
    multivariate_sym_hyperbolic_base_gen):
    """Multivariate Symmetric Marginal Hyperbolic model."""
    def _init_lamb(self, *args):
        self._lamb: float = 1.0


class multivariate_sym_hyperbolic_gen(multivariate_sym_hyperbolic_base_gen):
    """Multivariate Symmetric Hyperbolic model."""
    def _init_lamb(self, d: int):
        self._lamb: float = 0.5 * (d + 1)


class multivariate_sym_nig_gen(multivariate_sym_hyperbolic_base_gen):
    """Multivariate Symmetric Normal-Inverse Gaussian (NIG) model."""
    def _init_lamb(self, *args):
        self._lamb: float = -0.5

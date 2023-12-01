# Contains code for the multivariate symmetric Generalized Hyperbolic model
import numpy as np
from typing import Tuple, Union

from sklarpy.multivariate._prefit_dists import PreFitContinuousMultivariate
from sklarpy.multivariate._distributions._generalized_hyperbolic import \
    multivariate_gen_hyperbolic_gen
from sklarpy.utils._params import Params

__all__ = ['multivariate_sym_gen_hyperbolic_gen']


class multivariate_sym_gen_hyperbolic_gen(multivariate_gen_hyperbolic_gen):
    """Multivariate Symmetric Generalized Hyperbolic model."""
    _ASYMMETRIC = False

    def _check_params(self, params: tuple, **kwargs) -> None:
        # checking correct number of params passed
        if len(params) not in (5, 6):
            raise ValueError("Incorrect number of params given by user")

        # checking lambda, chi and psi
        self._check_w_params(params)

        # checking valid location vector and shape matrix
        loc, shape = params[3:5]
        definiteness, ones = (kwargs.get('definiteness', 'pd'),
                              kwargs.get('ones', False))
        self._check_loc_shape(loc, shape, definiteness, ones)

    def _get_params(self, params: Union[Params, tuple], **kwargs) -> tuple:
        params_tuple: tuple = PreFitContinuousMultivariate._get_params(
            self, params, **kwargs)
        loc: np.ndarray = params_tuple[3]
        gamma: np.ndarray = np.zeros(loc.shape, dtype=float)
        return *params_tuple[:5], gamma

    def _gh_to_params(self, params: tuple) -> tuple:
        return params[:-1]

    def _get_bounds(self, data: np.ndarray, as_tuple: bool = True, **kwargs) \
            -> Union[dict, tuple]:
        bounds_dict: dict = super()._get_bounds(data, False, **kwargs)
        # removing gamma from bounds
        return self._remove_bounds(bounds_dict, ['gamma'], data.shape[1],
                                   as_tuple)

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
        return np.array([*params[:3]], dtype=float)

    def _fit_given_params_tuple(self, params: tuple, **kwargs) \
            -> Tuple[dict, int]:
        self._check_params(params)
        return {'lamb': params[0], 'chi': params[1], 'psi': params[2],
                'loc': params[3], 'shape': params[4]}, params[3].size



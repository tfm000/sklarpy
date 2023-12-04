# Contains code for the multivariate Skewed-T model
import numpy as np
import scipy.special
from typing import Tuple, Union
from collections import deque
from scipy.optimize import differential_evolution

from sklarpy.multivariate._distributions._generalized_hyperbolic import \
    multivariate_gen_hyperbolic_gen
from sklarpy.multivariate._distributions._student_t import \
    multivariate_student_t_gen
from sklarpy.utils._params import Params
from sklarpy.misc import kv
from sklarpy.multivariate._prefit_dists import PreFitContinuousMultivariate
from sklarpy.univariate import ig
from sklarpy.univariate._distributions import _skewed_t

__all__ = ['multivariate_skewed_t_gen']


class multivariate_skewed_t_gen(multivariate_gen_hyperbolic_gen):
    """Multivariate Skewed-T model."""
    _NUM_W_PARAMS: int = 1
    _UNIVAR = _skewed_t

    def __init__(self, name: str, params_obj: Params, num_params: int,
                 max_num_variables: int, mvt_t: multivariate_student_t_gen):
        """A pre-fit continuous multivariate model.

        Parameters
        ----------
        name : str
            The name of the multivariate object.
            Used when saving, if a file path is not specified and/or for
            additional identification purposes.
        params_obj: Params
            The SklarPy Params object associated with this specific
            multivariate probability distribution.
        num_params: int
            The number of parameters which define this distribution.
        max_num_variables: int
            The maximum number of variables this multivariate probability
            distribution is defined for.
            i.e. 2 for bivariate models.
        mvt_t: multivariate_student_t_gen
            The multivariate student-t model.
        """
        super().__init__(
            name=name, params_obj=params_obj,
            num_params=num_params, max_num_variables=max_num_variables)
        self._mvt_t: multivariate_student_t_gen = mvt_t

    def _get_params(self, params: Union[Params, tuple], **kwargs) -> tuple:
        params_tuple: tuple = PreFitContinuousMultivariate._get_params(
            self, params, **kwargs)
        dof: float = params_tuple[1] if len(params_tuple) == 6 \
            else params_tuple[0]
        return -0.5 * dof, dof, 0, *params_tuple[-3:]

    def _check_w_params(self, params: tuple) -> None:
        # checking dof
        dof = params[1]
        dof_msg: str = 'dof must be a positive scalar'
        if not (isinstance(dof, float) or isinstance(dof, int)):
            raise TypeError(dof)
        elif dof <= 0:
            raise ValueError(dof_msg)

    def _check_params(self, params: tuple, **kwargs) -> None:
        # adjusting params to fit multivariate generalized hyperbolic params
        num_params: int = len(params)
        if num_params == 4:
            params = self._get_params(params, check_params=False)
        elif num_params != 6:
            raise ValueError("Incorrect number of params given by user")
        self._num_params = 6

        # checking params
        super()._check_params(params)
        self._num_params = 4

    def _singular_logpdf(self, xrow: np.ndarray, params: tuple, **kwargs
                         ) -> float:
        # getting params
        _, dof, _, loc, shape, gamma = params

        # reshaping for matrix multiplication
        d: int = loc.size
        loc = loc.reshape((d, 1))
        gamma = gamma.reshape((d, 1))
        xrow = xrow.reshape((d, 1))

        # common calculations
        shape_inv: np.ndarray = np.linalg.inv(shape)
        q: float = dof + ((xrow - loc).T @ shape_inv @ (xrow - loc))
        p: float = (gamma.T @ shape_inv @ gamma)
        s: float = 0.5*(dof + d)

        log_c: float = (1 - s) * np.log(2) - 0.5 * (
                2 * scipy.special.loggamma(0.5 * dof)
                + d * np.log(np.pi * dof) + np.log(np.linalg.det(shape)))
        log_h: float = kv.logkv(s, np.sqrt(q * p)) \
                       + (xrow - loc).T @ shape_inv @ gamma \
                       - s * (np.log(q / dof) - np.log(np.sqrt(q * p)))
        return float(log_c + log_h)

    def _logpdf_cdf(self, func_str: str, x: np.ndarray, params: tuple, **kwargs
                    ) -> np.ndarray:
        """Utility function able to implement logpdf and cdf methods without
        duplicate code.

        Parameters
        ----------
        func_str: str
            The name of the method to implement.
        x: Union[pd.DataFrame, np.ndarray]
            Our input data for our functions of the multivariate distribution.
        params: Union[pd.DataFrame, tuple]
            The parameters which define the multivariate model. These can be a
            Params object of the specific multivariate distribution or a tuple
            containing these parameters in the correct order.
        kwargs:
            kwargs to pass to the implemented method.

        Returns
        -------
        output: np.ndarray
            Implemented method values.
        """
        if not np.any(params[-1]):
            # gamma is an array of 0's, hence we use the symmetric,
            # multivariate student-T distribution.
            params: tuple = (params[1], *params[3:5])
            obj = self._mvt_t
        else:
            obj = super()
        return eval(f'obj._{func_str}(x=x, params=params, **kwargs)')

    def _logpdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        return self._logpdf_cdf(func_str='logpdf', x=x, params=params,
                                **kwargs)

    def _cdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        return self._logpdf_cdf(func_str='cdf', x=x, params=params, **kwargs)

    def _w_rvs(self, size: int, params: tuple) -> np.ndarray:
        alpha_beta: float = params[1] / 2
        return ig.rvs((size, ), (alpha_beta, alpha_beta), ppf_approx=True)

    def _get_bounds(self, data: np.ndarray, as_tuple: bool = True, **kwargs
                    ) -> Union[dict, tuple]:
        bounds = super()._get_bounds(data, as_tuple, **kwargs)

        # removing lambda and psi from bounds
        if as_tuple:
            dof_bounds: tuple = bounds[1]
            bounds = (dof_bounds, *bounds[3:])
        else:
            bounds.pop('lamb')
            bounds.pop('psi')
            bounds['dof'] = bounds.pop('chi')
        return bounds

    def _etas_deltas_zetas(self, data: np.ndarray, params: tuple, h: float
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, dof, _, loc, shape, gamma = params
        shape_inv: np.ndarray = np.linalg.inv(shape)
        d: int = loc.size
        p: float = float(gamma.T @ shape_inv @ gamma)
        s: float = 0.5 * (dof + d)
        etas: deque = deque()
        deltas: deque = deque()
        zetas: deque = deque()

        for xi in data:
            xi = xi.reshape((d, 1))
            qi: float = float(dof + ((xi - loc).T @ shape_inv @ (xi - loc)))

            cond_params: tuple = (-s, qi, p)
            eta_i: float = multivariate_gen_hyperbolic_gen._UNIVAR._exp_w(
                cond_params)
            delta_i: float = multivariate_gen_hyperbolic_gen._UNIVAR._exp_w(
                (s, p, qi))
            zeta_i: float = multivariate_gen_hyperbolic_gen._exp_log_w(
                cond_params, h)

            deltas.append(delta_i)
            etas.append(eta_i)
            zetas.append(zeta_i)

        n: int = len(etas)
        return (np.asarray(etas).reshape((n, 1)),
                np.asarray(deltas).reshape((n, 1)),
                np.asarray(zetas).reshape((n, 1)))

    def _add_randomness(self, params: tuple, bounds: tuple, d: int,
                        randomness_var: float, copula: bool) -> tuple:
        adj_params: tuple = super()._add_randomness(
            params=params, bounds=bounds, d=d,
            randomness_var=randomness_var, copula=copula)
        return self._get_params(adj_params, check_params=False)

    def _neg_q2(self, dof: float, etas: np.ndarray, deltas: np.ndarray,
                zetas: np.ndarray) -> float:
        delta_mean, zeta_mean = deltas.mean(), zetas.mean()
        val: float = -scipy.special.digamma(dof / 2) + np.log(dof / 2) + 1 \
                     - zeta_mean - delta_mean
        return abs(val)

    def _q2_opt(self, bounds: tuple, etas: np.ndarray, deltas: np.ndarray,
                zetas: np.ndarray, q2_options: dict):
        q2_res = differential_evolution(
            self._neg_q2, bounds=(bounds[0], ),
            args=(etas, deltas, zetas), **q2_options)
        dof: float = float(q2_res['x'])
        return {'x': np.array([-0.5 * dof, dof, 0.0], dtype=float),
                'success': q2_res['success']}

    def _gh_to_params(self, params: tuple) -> tuple:
        return params[1], *params[3:]

    def _theta_to_params(self, theta: np.ndarray, mean: np.ndarray,
                         S: np.ndarray, S_det: float, min_eig: float,
                         copula: bool, **kwargs) -> tuple:
        # modifying theta to fit that of the Generalized Hyperbolic
        dof: float = theta[0]
        theta = np.array([-dof / 2, dof, 0, *theta[1:]])
        return super()._theta_to_params(theta=theta, mean=mean, S=S,
                                        S_det=S_det, min_eig=min_eig,
                                        copula=copula, **kwargs)

    def _params_to_theta(self, params: tuple, **kwargs) -> np.ndarray:
        return np.array([params[1], *params[-1].flatten()], dtype=float)

    def _get_params0(self, data: np.ndarray, bounds: tuple, cov_method: str,
                     min_eig, copula: bool, **kwargs) -> tuple:
        # modifying bounds to fit those of the Generalized Hyperbolic
        bounds = ((0, 0), bounds[0], (0, 0), *bounds[1:])
        params0: tuple = super()._get_params0(
            data=data, bounds=bounds, cov_method=cov_method,
            min_eig=min_eig, copula=copula, **kwargs)

        if not kwargs.get('em_opt', False):
            return params0

        # EM algorithm requires gamma parameter to be non-zero
        d: int = data.shape[1]
        data_stds: np.ndarray = data.std(axis=0, dtype=float).reshape((d, 1))
        gamma: np.ndarray = np.array([0])
        while not np.any(gamma):
            gamma = np.random.normal(scale=data_stds, size=(d, 1))
        return (*params0[:-1], gamma)

    def _fit_given_params_tuple(self, params: tuple, **kwargs
                                ) -> Tuple[dict, int]:
        self._check_params(params, **kwargs)
        return {'dof': params[0], 'loc': params[1], 'shape': params[2],
                'gamma': params[3]}, params[1].size

# Contains code for the multivariate Skewed-T model
import numpy as np
import scipy.special
from typing import Tuple, Union
from collections import deque
from scipy.optimize import differential_evolution

from sklarpy.multivariate._distributions._generalized_hyperbolic import \
    multivariate_gen_hyperbolic_gen
from sklarpy._other import Params
from sklarpy.misc import kv
from sklarpy.multivariate._prefit_dists import PreFitContinuousMultivariate
from sklarpy.univariate import ig

__all__ = ['multivariate_skewed_t_gen']


class multivariate_skewed_t_gen(multivariate_gen_hyperbolic_gen):
    _NUM_W_PARAMS: int = 1

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
            eta_i: float = multivariate_gen_hyperbolic_gen._exp_w(cond_params)
            delta_i: float = multivariate_gen_hyperbolic_gen._exp_w((s, p, qi))
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

    def _get_params0(self, data: np.ndarray, bounds: tuple, copula: bool,
                     **kwargs) -> tuple:
        # getting theta0
        d: int = data.shape[1]
        dof0: float = np.random.uniform(*bounds[0])
        data_stds: np.ndarray = data.std(axis=0, dtype=float)
        gamma0: np.ndarray = np.random.normal(scale=data_stds, size=(d,))
        if not copula:
            loc0: np.ndarray = data.mean(axis=0, dtype=float).flatten()
            theta0: np.ndarray = np.array([dof0, *loc0, *gamma0], dtype=float)
        else:
            theta0: np.ndarray = np.array([dof0, *gamma0], dtype=float)

        # converting to params0
        S, S_det, min_eig, copula = super(
        )._get_low_dim_mle_objective_func_args(
            data=data, copula=copula,
            cov_method=kwargs.get('cov_method', 'pp_kendall'), min_eig=None)

        return self._low_dim_theta_to_params(theta=theta0, S=S, S_det=S_det,
                                             min_eig=min_eig, copula=copula)

    @staticmethod
    def _exp_w(params: tuple) -> float:
        alpha_beta: float = params[1] / 2
        return alpha_beta / (alpha_beta - 1)

    @staticmethod
    def _var_w(params: tuple) -> float:
        alpha_beta: float = params[1] / 2
        return (alpha_beta ** 2) / (((alpha_beta - 1)**2) * (alpha_beta-2))

    def _low_dim_theta_to_params(self, theta: np.ndarray, S: np.ndarray,
                                 S_det: float, min_eig: float, copula: bool
                                 ) -> tuple:
        dof: float = theta[0]
        theta = np.array([-dof/2, dof, 0.0, *theta[1:]], dtype=float)
        _, dof, _, loc, shape, gamma = super()._low_dim_theta_to_params(
            theta=theta, S=S, S_det=S_det, min_eig=min_eig, copula=copula)
        return dof, loc, shape, gamma

    def _params_to_low_dim_theta(self, params: tuple, copula: bool
                                 ) -> np.ndarray:
        params: tuple = self._gh_to_params(params)
        if copula:
            return np.array([params[0], *params[-1].flatten()], dtype=float)
        return np.array([params[0], *params[1].flatten(),
                         *params[-1].flatten()], dtype=float)

    def _fit_given_params_tuple(self, params: tuple, **kwargs
                                ) -> Tuple[dict, int]:
        self._check_params(params, **kwargs)
        return {'dof': params[0], 'loc': params[1], 'shape': params[2],
                'gamma': params[3]}, params[1].size

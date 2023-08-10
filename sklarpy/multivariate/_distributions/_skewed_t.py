import numpy as np
import scipy.special
from typing import Tuple, Union
from collections import deque
from scipy.optimize import differential_evolution

from sklarpy.multivariate._distributions._generalized_hyperbolic import multivariate_gen_hyperbolic_gen
from sklarpy._other import Params
from sklarpy.misc import kv
from sklarpy.multivariate._prefit_dists import PreFitContinuousMultivariate
from sklarpy.univariate import ig

__all__ = ['multivariate_skewed_t_gen']


class multivariate_skewed_t_gen(multivariate_gen_hyperbolic_gen):
    def _get_params(self, params: Union[Params, tuple], **kwargs) -> tuple:
        params_tuple: tuple = PreFitContinuousMultivariate._get_params(self, params, **kwargs)
        dof: float = params_tuple[1] if len(params_tuple) == 6 else params_tuple[0]
        return -0.5*dof, dof, 0, *params_tuple[-3:]

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

    def _singular_logpdf(self, x: np.ndarray, params: tuple, **kwargs) -> float:
        # getting params
        _, dof, _, loc, shape, gamma = params

        # reshaping for matrix multiplication
        d: int = loc.size
        loc = loc.reshape((d, 1))
        gamma = gamma.reshape((d, 1))
        x = x.reshape((d, 1))

        # common calculations
        shape_inv: np.ndarray = np.linalg.inv(shape)
        q: float = dof + ((x - loc).T @ shape_inv @ (x - loc))
        p: float = (gamma.T @ shape_inv @ gamma)
        s: float = 0.5*(dof + d)

        log_c: float = (1 - s) * np.log(2) - 0.5 * (2 * scipy.special.loggamma(0.5 * dof) + d * np.log(np.pi * dof) + np.log(np.linalg.det(shape)))
        log_h: float = kv.logkv(s, np.sqrt(q * p)) + (x - loc).T @ shape_inv @ gamma - s * (np.log(q / dof) - np.log(np.sqrt(q * p)))
        return float(log_c + log_h)

    def _w_rvs(self, size: int, params: tuple) -> np.ndarray:
        alpha_beta: float = params[1] / 2
        return ig.rvs((size, ), (alpha_beta, alpha_beta), ppf_approx=True)

    def _get_bounds(self, data: np.ndarray, as_tuple: bool = True, **kwargs) -> Union[dict, tuple]:
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

    def _etas_deltas_zetas(self, data: np.ndarray, params: tuple, h: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            zeta_i: float = multivariate_gen_hyperbolic_gen._exp_log_w(cond_params, h)

            deltas.append(delta_i)
            etas.append(eta_i)
            zetas.append(zeta_i)

        n: int = len(etas)
        return np.asarray(etas).reshape((n, 1)), np.asarray(deltas).reshape((n, 1)), np.asarray(zetas).reshape((n, 1))

    def _add_randomness(self, params: tuple, bounds: tuple, d: int, randomness_var: float) -> tuple:
        adj_params: tuple = super()._add_randomness(params, bounds, d, randomness_var)
        return self._get_params(adj_params, check_params=False)

    def _neg_q2(self, dof: float, etas: np.ndarray, deltas: np.ndarray, zetas: np.ndarray) -> float:
        delta_mean, zeta_mean = deltas.mean(), zetas.mean()
        val: float = -scipy.special.digamma(dof / 2) + np.log(dof / 2) + 1 - zeta_mean - delta_mean
        return abs(val)

    def _q2_opt(self, bounds: tuple, etas: np.ndarray, deltas: np.ndarray, zetas: np.ndarray, q2_options: dict):
        q2_res = differential_evolution(self._neg_q2, bounds=(bounds[0], ), args=(etas, deltas, zetas), **q2_options)
        dof: float = float(q2_res['x'])
        return {'x': np.array([-0.5 * dof, dof, 0.0], dtype=float), 'success': q2_res['success']}

    def _gh_to_params(self, params: tuple) -> tuple:
        return params[1], *params[3:]

    def _get_low_dim_theta0(self, data: np.ndarray, bounds: tuple, copula: bool=False) -> np.ndarray:
        d: int = data.shape[1]
        dof0: float = np.random.uniform(*bounds[0])
        if not copula:
            loc0: np.ndarray = data.mean(axis=0, dtype=float).flatten()
        else:
            loc0: np.ndarray = np.zeros((d,), dtype=float)
        data_stds: np.ndarray = data.std(axis=0, dtype=float)
        gamma0: np.ndarray = np.random.normal(scale=data_stds, size=(d,))
        theta0: np.ndarray = np.array([dof0, *loc0, *gamma0], dtype=float)
        return theta0

    @staticmethod
    def _exp_w(params: tuple) -> float:
        alpha_beta: float = params[1] / 2
        return alpha_beta / (alpha_beta - 1)

    @staticmethod
    def _var_w(params: tuple) -> float:
        alpha_beta: float = params[1] / 2
        return (alpha_beta ** 2) / (((alpha_beta - 1)**2) * (alpha_beta-2))

    def _low_dim_theta_to_params(self, theta: np.ndarray, S: np.ndarray, S_det: float, min_eig: float, copula: bool = False) -> tuple:
        dof: float = theta[0]
        theta = np.array([-dof/2, dof, 0.0, *theta[1:]], dtype=float)
        _, dof, _, loc, shape, gamma = super()._low_dim_theta_to_params(theta=theta, S=S, S_det=S_det, min_eig=min_eig, copula=copula)
        return dof, loc, shape, gamma

    def _fit_given_params_tuple(self, params: tuple, **kwargs) -> Tuple[dict, int]:
        self._check_params(params, **kwargs)
        return {'dof': params[0], 'loc': params[1], 'shape': params[2], 'gamma': params[3]}, params[1].size





# if __name__ == '__main__':
#     my_dof = 10.0
#     my_loc = np.array([2.19459261, -5.03119294], dtype=float)
#     my_shape = np.array([[ 2.53039133, -1.50221816], [-1.50221816,  5.49204651]], dtype=float)
#     # my_gamma = np.array([2.3, -4.3], dtype=float)
#     my_gamma = np.array([0.81390385, -1.7244502], dtype=float)
#     my_params = (my_dof, my_loc, my_shape, my_gamma)
#     my_params2 = (4.5, *my_params[1:])
#
#     dist = multivariate_skewed_t
#     dist.pdf_plot(params=my_params)
#     # breakpoint()
#
#     # rvs = dist.rvs(1000, my_params)
#     import pandas as pd
#     rvs = pd.read_excel('h_rvs.xlsx', index_col=0)
#     # # rvs = rvs.to_numpy()
#     # # print(rvs)
#     # # my_dist = dist.fit(rvs, show_progress=True, min_retries=1, max_retries=1, tol=0.1)
#     my_dist = dist.fit(rvs, show_progress=True, method='em', min_retries=0, max_retries=3, tol=0.1)
#     print('theoretical max: ', dist.loglikelihood(rvs, my_params))
#     print(my_dist.params.to_dict)
#     #
#     my_dist.pdf_plot()
#     #
#     # import matplotlib.pyplot as plt
#     # #
#     # # my_dist.pdf_plot(show=False)
#     # # my_dist.mc_cdf_plot(show=False)
#     # #
#     # p1 = dist.pdf(rvs, my_params)
#     # p2 = my_dist.pdf(rvs)
#     # fig = plt.figure()
#     # ax = fig.add_subplot(projection='3d')
#     # num = rvs.shape[0]
#     # ax.scatter(rvs[:num, 0], rvs[:num, 1], p1[:num], marker='o', c='r')
#     # ax.scatter(rvs[:num, 0], rvs[:num, 1], p2[:num], marker='^', c='b')
#     # plt.show()
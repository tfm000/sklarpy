import numpy as np
from typing import Tuple, Union
from scipy.optimize import differential_evolution
from abc import abstractmethod

from sklarpy.multivariate._distributions._generalized_hyperbolic import multivariate_gen_hyperbolic_gen
from sklarpy.multivariate._prefit_dists import PreFitContinuousMultivariate
from sklarpy._other import Params

__all__ = ['multivariate_marginal_hyperbolic_gen', 'multivariate_hyperbolic_gen', 'multivariate_nig_gen']


class multivariate_hyperbolic_base_gen(multivariate_gen_hyperbolic_gen):
    _NUM_W_PARAMS: int = 2

    @abstractmethod
    def _init_lamb(self, *args):
        pass

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
        params_tuple: tuple = PreFitContinuousMultivariate._get_params(self, params, **kwargs)
        self._init_lamb(params_tuple[-1].shape[0])
        params_tuple = self._lamb, *params_tuple
        return params_tuple[-6:]

    def _get_bounds(self, data: np.ndarray, as_tuple: bool = True, **kwargs) -> Union[dict, tuple]:
        bounds_dict: dict = super()._get_bounds(data, False, **kwargs)
        # removing lambda from bounds
        return self._remove_bounds(bounds_dict, ['lamb'], data.shape[1], as_tuple)

    def _add_randomness(self, params: tuple, bounds: tuple, d: int, randomness_var: float, copula) -> tuple:
        adj_params: tuple = super()._add_randomness(params, bounds, d, randomness_var, copula)
        return self._get_params(adj_params, check_params=False)

    def _neg_q2(self, w_params: np.ndarray, etas: np.ndarray, deltas: np.ndarray, zetas: np.ndarray) -> float:
        return super()._neg_q2((self._lamb, *w_params), etas, deltas, zetas)

    def _q2_opt(self, bounds: tuple, etas: np.ndarray, deltas: np.ndarray, zetas: np.ndarray, q2_options: dict):
        q2_res = differential_evolution(self._neg_q2, bounds=bounds[:2], args=(etas, deltas, zetas), **q2_options)
        chi, psi = q2_res['x']
        return {'x': np.array([self._lamb, chi, psi]), 'success': q2_res['success']}

    def _gh_to_params(self, params: tuple) -> tuple:
        return params[1:]

    def _get_low_dim_theta0(self, data: np.ndarray, bounds: tuple, copula: bool) -> np.ndarray:
        d: int = (len(bounds) - 2) / 2 if self._ASYMMETRIC else len(bounds) - 2
        d = int(d)
        self._init_lamb(d)
        bounds = ((self._lamb, self._lamb), *bounds)
        theta0: np.ndarray = super()._get_low_dim_theta0(data=data, bounds=bounds, copula=copula)
        return theta0[1:]

    def _low_dim_theta_to_params(self, theta: np.ndarray, S: np.ndarray, S_det: float, min_eig: float, copula: bool) -> tuple:
        theta = np.array([self._lamb, *theta], dtype=float)
        params: tuple = super()._low_dim_theta_to_params(theta=theta, S=S, S_det=S_det, min_eig=min_eig, copula=copula)
        return params[1:]

    def _fit_given_params_tuple(self, params: tuple, **kwargs) -> Tuple[dict, int]:
        self._check_params(params, **kwargs)
        return {'chi': params[0], 'psi': params[1], 'loc': params[2], 'shape': params[3], 'gamma': params[4]}, params[2].size


class multivariate_marginal_hyperbolic_gen(multivariate_hyperbolic_base_gen):
    def _init_lamb(self, *args):
        self._lamb: float = 1.0


class multivariate_hyperbolic_gen(multivariate_hyperbolic_base_gen):
    def _init_lamb(self, d: int):
        self._lamb: float = 0.5 * (d + 1)


class multivariate_nig_gen(multivariate_hyperbolic_base_gen):
    def _init_lamb(self, *args):
        self._lamb: float = -0.5



# if __name__ == '__main__':
#     # my_loc = np.array([1, -3, 5.2], dtype=float)
#     # my_shape = np.array([[1, 0.284, 0.520], [0.284, 1, 0.435], [0.520, 0.435, 1]], dtype=float)
#     # my_gamma = np.array([2.3, 1.4, -4.3], dtype=float)
#     # my_chi = 1.7
#     # my_psi = 4.5
#
#     my_loc = np.array([1, -3], dtype=float)
#     my_shape = np.array([[1, 0.7], [0.7, 1]], dtype=float)
#     my_chi = 1.7
#     my_psi = 4.5
#     my_gamma = np.array([2.3, -4.3], dtype=float)
#
#     dist = multivariate_marginal_hyperbolic
#     # dist = multivariate_hyperbolic
#     # dist = multivariate_nig
#
#     my_params = (my_chi, my_psi, my_loc, my_shape, my_gamma)
#
#     rvs = dist.rvs(1000, my_params)
#
#     # import pandas as pd
#     #
#     # pd.DataFrame(rvs).to_excel('h_rvs.xlsx')
#     # print(rvs)
#     # multivariate_marginal_hyperbolic.pdf_plot(params=my_params)
#     # print(multivariate_marginal_hyperbolic.pdf(rvs, my_params))
#
#     my_dist = dist.fit(rvs, show_progress=True, min_retries=1, max_retries=1, tol=0.1, copula=True)
#     # my_dist = dist.fit(rvs, show_progress=True, method='em', min_retries=1, max_retries=1, tol=0.1)
#     print('theoretical max: ', dist.loglikelihood(rvs, my_params))
#     print(my_dist.params.to_dict)
#
#     import matplotlib.pyplot as plt
#
#     p1 = dist.pdf(rvs, my_params)
#     p2 = my_dist.pdf(rvs)
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     num = rvs.shape[0]
#     ax.scatter(rvs[:num, 0], rvs[:num, 1], p1[:num], marker='o', c='r')
#     ax.scatter(rvs[:num, 0], rvs[:num, 1], p2[:num], marker='^', c='b')
#     plt.show()
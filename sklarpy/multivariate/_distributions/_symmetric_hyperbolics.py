import numpy as np
from typing import Tuple, Union

from sklarpy.multivariate._distributions._hyperbolics import multivariate_hyperbolic_base_gen
from sklarpy.multivariate._prefit_dists import PreFitContinuousMultivariate
from sklarpy._other import Params

__all__ = ['multivariate_sym_marginal_hyperbolic_gen']


class multivariate_sym_hyperbolic_base_gen(multivariate_hyperbolic_base_gen):
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
        params_tuple: tuple = PreFitContinuousMultivariate._get_params(self, params, **kwargs)
        self._init_lamb(params_tuple[-1].shape[0])
        if len(params_tuple) == 6:
            return params_tuple

        # 4 params
        loc: np.ndarray = params_tuple[2]
        gamma: np.ndarray = np.zeros(loc.shape, dtype=float)
        return self._lamb, *params_tuple, gamma

    def _get_bounds(self, data: np.ndarray, as_tuple: bool = True, **kwargs) -> Union[dict, tuple]:
        bounds_dict: dict = super()._get_bounds(data, False, **kwargs)
        # removing gamma from bounds
        return self._remove_bounds(bounds_dict, ['gamma'], data.shape[1], as_tuple)

    def _gh_to_params(self, params: tuple) -> tuple:
        return params[1:5]

    def _get_low_dim_theta0(self, data: np.ndarray, bounds: tuple, copula: bool) -> np.ndarray:
        theta0: np.ndarray = super()._get_low_dim_theta0(data, bounds, copula)
        d: int = data.shape[1]
        return theta0[:-d]

    def _low_dim_theta_to_params(self, theta: np.ndarray, S: np.ndarray, S_det: float, min_eig: float, copula: bool) -> tuple:
        d: int = S.shape[0]

        chi, psi = theta[:2]

        if not copula:
            # getting location vector from theta
            loc: np.ndarray = theta[2:].copy()
            loc = loc.reshape((d, 1))

            # calculating implied shape parameter
            exp_w: float = self._exp_w_a((self._lamb, chi, psi), 1)
            shape: np.ndarray = S / exp_w
        else:
            loc: np.ndarray = np.zeros((d, 1), dtype=float)
            shape: np.ndarray = S

        return chi, psi, loc, shape

    def _fit_given_params_tuple(self, params: tuple, **kwargs) -> Tuple[dict, int]:
        self._check_params(params, **kwargs)
        return {'chi': params[0], 'psi': params[1], 'loc': params[2], 'shape': params[3]}, params[2].size


class multivariate_sym_marginal_hyperbolic_gen(multivariate_sym_hyperbolic_base_gen):
    def _init_lamb(self, *args):
        self._lamb: float = 1.0


class multivariate_sym_hyperbolic_gen(multivariate_sym_hyperbolic_base_gen):
    def _init_lamb(self, d: int):
        self._lamb: float = 0.5 * (d + 1)


class multivariate_sym_nig_gen(multivariate_sym_hyperbolic_base_gen):
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
#
#     # dist = multivariate_sym_marginal_hyperbolic
#     dist = multivariate_sym_hyperbolic
#     # dist = multivariate_sym_nig
#
#     my_params = (my_chi, my_psi, my_loc, my_shape)
#
#     rvs = dist.rvs(1000, my_params)
#     # print(rvs)
#     # multivariate_marginal_hyperbolic.pdf_plot(params=my_params)
#     # print(multivariate_marginal_hyperbolic.pdf(rvs, my_params))
#
#     # my_dist = dist.fit(rvs, show_progress=True, min_retries=1, max_retries=1, tol=0.1, copula=True)
#     my_dist = dist.fit(rvs, show_progress=True, min_retries=1, max_retries=1, tol=0.1, method='em', copula=True)
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
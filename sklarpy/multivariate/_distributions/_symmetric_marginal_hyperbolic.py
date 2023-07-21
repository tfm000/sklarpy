import numpy as np
from typing import Tuple, Union

from sklarpy.multivariate._distributions._marginal_hyperbolic import multivariate_marginal_hyperbolic_gen
from sklarpy.multivariate._prefit_dists import PreFitContinuousMultivariate
from sklarpy._other import Params
from sklarpy.multivariate._distributions._params import MultivariateSymMarginalHyperbolicParams


__all__ = ['multivariate_sym_marginal_hyperbolic']


class multivariate_sym_marginal_hyperbolic_gen(multivariate_marginal_hyperbolic_gen):
    _ASYMMETRIC = False

    def _check_params(self, params: tuple, **kwargs) -> None:
        # adjusting params to fit multivariate generalized hyperbolic params
        if len(params) == 4:
            params = self._get_params(params, check_params=False)
        elif len(params) != 6:
            raise ValueError("Incorrect number of params given by user")
        self._num_params = 6

        # checking params
        super()._check_params(params)
        self._num_params = 4

    def _get_params(self, params: Union[Params, tuple], **kwargs) -> tuple:
        params_tuple: tuple = PreFitContinuousMultivariate._get_params(self, params, **kwargs)
        if len(params_tuple) == 6:
            return params_tuple

        # 4 params
        loc: np.ndarray = params_tuple[2]
        gamma: np.ndarray = np.zeros(loc.shape, dtype=float)
        return 1.0, *params_tuple, gamma

    def _get_bounds(self, data: np.ndarray, as_tuple: bool = True, **kwargs) -> Union[dict, tuple]:
        bounds = super()._get_bounds(data, as_tuple, **kwargs)

        # removing gamma from bounds
        if as_tuple:
            d: int = data.shape[1]
            bounds = bounds[:-d]
        else:
            bounds.pop('gamma')
        return bounds

    def _gh_to_params(self, params: tuple) -> tuple:
        return params[1:5]

    def _get_low_dim_theta0(self, data: np.ndarray, bounds: tuple) -> np.ndarray:
        theta0: np.ndarray = super()._get_low_dim_theta0(data, bounds)
        d: int = data.shape[1]
        return theta0[:-d]

    def _low_dim_theta_to_params(self, theta: np.ndarray, S: np.ndarray, S_det: float) -> tuple:
        d: int = S.shape[0]

        chi, psi = theta[:2]
        loc: np.ndarray = theta[2:].copy()
        loc = loc.reshape((d, 1))

        exp_w: float = self._exp_w_a((1, chi, psi), 1)
        shape: np.ndarray = S / exp_w
        return chi, psi, loc, shape

    def _fit_given_params_tuple(self, params: tuple, **kwargs) -> Tuple[dict, int]:
        self._check_params(params, **kwargs)
        return {'chi': params[0], 'psi': params[1], 'loc': params[2], 'shape': params[3]}, params[2].size


multivariate_sym_marginal_hyperbolic: multivariate_sym_marginal_hyperbolic_gen = multivariate_sym_marginal_hyperbolic_gen(name='multivariate_sym_marginal_hyperbolic', params_obj=MultivariateSymMarginalHyperbolicParams, num_params=4, max_num_variables=np.inf)


if __name__ == '__main__':
    # my_loc = np.array([1, -3, 5.2], dtype=float)
    # my_shape = np.array([[1, 0.284, 0.520], [0.284, 1, 0.435], [0.520, 0.435, 1]], dtype=float)
    # my_gamma = np.array([2.3, 1.4, -4.3], dtype=float)
    # my_chi = 1.7
    # my_psi = 4.5

    my_loc = np.array([1, -3], dtype=float)
    my_shape = np.array([[1, 0.7], [0.7, 1]], dtype=float)
    my_chi = 1.7
    my_psi = 4.5

    my_params = (my_chi, my_psi, my_loc, my_shape)

    rvs = multivariate_sym_marginal_hyperbolic.rvs(1000, my_params)
    # print(rvs)

    my_sym_marg_hyperbolic = multivariate_sym_marginal_hyperbolic.fit(rvs, method='em', show_progress=True, min_retries=1, max_retries=1)
    print('theoretical max: ', multivariate_sym_marginal_hyperbolic.loglikelihood(rvs, my_params))
    print(my_sym_marg_hyperbolic.params.to_dict)

    import matplotlib.pyplot as plt

    p1 = multivariate_sym_marginal_hyperbolic.pdf(rvs, my_params)
    p2 = my_sym_marg_hyperbolic.pdf(rvs)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    num = rvs.shape[0]
    ax.scatter(rvs[:num, 0], rvs[:num, 1], p1[:num], marker='o', c='r')
    ax.scatter(rvs[:num, 0], rvs[:num, 1], p2[:num], marker='^', c='b')
    plt.show()
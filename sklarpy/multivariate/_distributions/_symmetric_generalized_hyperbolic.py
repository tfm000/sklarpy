import numpy as np
from typing import Tuple, Union

from sklarpy.multivariate._prefit_dists import PreFitContinuousMultivariate
from sklarpy.multivariate._distributions._generalized_hyperbolic import multivariate_gen_hyperbolic_gen
from sklarpy._other import Params

__all__ = ['multivariate_sym_gen_hyperbolic_gen']


class multivariate_sym_gen_hyperbolic_gen(multivariate_gen_hyperbolic_gen):
    _ASYMMETRIC = False

    def _check_params(self, params: tuple, **kwargs) -> None:
        # checking correct number of params passed
        if len(params) not in (5, 6):
            raise ValueError("Incorrect number of params given by user")

        # checking lambda, chi and psi
        self._check_w_params(params)

        # checking valid location vector and shape matrix
        loc, shape = params[3:5]
        definiteness, ones = kwargs.get('definiteness', 'pd'), kwargs.get('ones', False)
        self._check_loc_shape(loc, shape, definiteness, ones)

    def _get_params(self, params: Union[Params, tuple], **kwargs) -> tuple:
        params_tuple: tuple = PreFitContinuousMultivariate._get_params(self, params, **kwargs)
        loc: np.ndarray = params_tuple[3]
        gamma: np.ndarray = np.zeros(loc.shape, dtype=float)
        return *params_tuple[:5], gamma

    def _gh_to_params(self, params: tuple) -> tuple:
        return params[:-1]

    def _get_bounds(self, data: np.ndarray, as_tuple: bool = True, **kwargs) -> Union[dict, tuple]:
        bounds_dict: dict = super()._get_bounds(data, False, **kwargs)
        # removing gamma from bounds
        return self._remove_bounds(bounds_dict, ['gamma'], data.shape[1], as_tuple)

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

    def _fit_given_params_tuple(self, params: tuple, **kwargs) -> Tuple[dict, int]:
        self._check_params(params)
        return {'lamb': params[0], 'chi': params[1], 'psi': params[2], 'loc': params[3], 'shape': params[4]}, params[3].size


# multivariate_sym_gen_hyperbolic: multivariate_sym_gen_hyperbolic_gen = multivariate_sym_gen_hyperbolic_gen(name="multivariate_sym_gen_hyperbolic", params_obj=MultivariateSymGenHyperbolicParams, num_params=5, max_num_variables=np.inf)
#
#
# if __name__ == '__main__':
#     # my_loc = np.array([1, -3, 5.2], dtype=float)
#     # my_shape = np.array([[1, 0.284, 0.520], [0.284, 1, 0.435], [0.520, 0.435, 1]], dtype=float)
#     # my_lambda = - 0.5
#     # my_chi = 1.7
#     # my_psi = 4.5
#     # my_params = (my_lambda, my_chi, my_psi, my_loc, my_shape)
#
#     my_loc = np.array([1, -3], dtype=float)
#     my_shape = np.array([[1, 0.7], [0.7, 1]], dtype=float)
#     my_lambda = - 0.5
#     my_chi = 1.7
#     my_psi = 4.5
#     my_params = (my_lambda, my_chi, my_psi, my_loc, my_shape)
#
#     rvs = multivariate_sym_gen_hyperbolic.rvs(10000, my_params)
#     my_sym = multivariate_sym_gen_hyperbolic.fit(rvs, show_progress=True, copula=True)
#     print(my_sym.params.to_dict)
#     print('theoretical max: ', multivariate_sym_gen_hyperbolic.loglikelihood(rvs, my_params))
#     # rvs2 = my_sym.rvs(10000)
#     p1 = multivariate_sym_gen_hyperbolic.pdf(rvs, my_params)
#     p2 = my_sym.pdf(rvs)
#
#     multivariate_sym_gen_hyperbolic.pdf_plot(params=my_params, show=False)
#     import matplotlib.pyplot as plt
#
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     num=rvs.shape[0]
#     ax.scatter(rvs[:num, 0], rvs[:num, 1], p1[:num], marker='o', c='r')
#     ax.scatter(rvs[:num, 0], rvs[:num, 1], p2[:num], marker='^', c='b')
#     plt.show()
#     breakpoint()
#     # breakpoint()
#     # my_sym.pdf_plot()
#     # my_sym.mc_cdf_plot()
#     # my_sym.marginal_pairplot()
#     # my_sym.rvs()


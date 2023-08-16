# contains code for the multivariate student t distribution
import numpy as np
import scipy.stats
import scipy.integrate
import scipy.optimize
from typing import Tuple, Union

from sklarpy.multivariate._prefit_dists import PreFitContinuousMultivariate
from sklarpy.multivariate._fitted_dists import FittedContinuousMultivariate
from sklarpy._other import Params
from sklarpy._utils import dataframe_or_array

__all__ = ['multivariate_student_t_gen']


class multivariate_student_t_gen(PreFitContinuousMultivariate):
    _DATA_FIT_METHODS = (*PreFitContinuousMultivariate._DATA_FIT_METHODS, 'dof_low_dim_mle')

    def __cov_to_shape(self, A: np.ndarray, dof: float, reverse: bool = False) -> np.ndarray:
        scale: float = (dof - 2) / dof
        if reverse:
            return A / scale if dof > 2 else A
        return A * scale if dof > 2 else A

    def _check_params(self, params: tuple, **kwargs) -> None:
        # checking correct number of params passed
        super()._check_params(params)

        # checking valid location vector and shape matrix
        loc, shape, dof = params
        definiteness, ones = kwargs.get('definiteness', 'pd'), kwargs.get('ones', False)
        self._check_loc_shape(loc, shape, definiteness, ones)

        # checking valid dof parameter
        dof_msg: str = 'dof parameter must be a positive scalar'
        if not (isinstance(dof, float) or not isinstance(dof, int)):
            raise TypeError(dof_msg)
        elif dof <= 0:
            raise ValueError(dof_msg)

    def _logpdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        return np.array([scipy.stats.multivariate_t.logpdf(x, loc=params[0].flatten(), shape=params[1], df=params[2])], dtype=float).flatten()

    def _cdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        return np.array([scipy.stats.multivariate_t.cdf(x, loc=params[0].flatten(), shape=params[1], df=params[2])], dtype=float).flatten()

    def _rvs(self, size: int, params: tuple) -> np.ndarray:
        return scipy.stats.multivariate_t.rvs(size=size, loc=params[0].flatten(), shape=params[1], df=params[2])

    def _get_bounds(self, data: np.ndarray, as_tuple: bool, **kwargs) -> Union[dict, tuple]:
        d: int = data.shape[1]
        data_bounds: np.ndarray = np.array([data.min(axis=0), data.max(axis=0)], dtype=float).T
        default_bounds: dict = {'loc': data_bounds, 'dof': (2.01, 100.0)}
        return super()._get_bounds(default_bounds, d, as_tuple, **kwargs)

    def _get_low_dim_theta0(self, data: np.ndarray, bounds: tuple, copula: bool) -> np.ndarray:
        dof0: float = np.random.uniform(*bounds[-1])
        if not copula:
            loc0: np.ndarray = data.mean(axis=0, dtype=float).flatten()
            return np.array([*loc0, dof0], dtype=float)
        return np.array([dof0], dtype=float)

    def _low_dim_theta_to_params(self, theta: np.ndarray, S: np.ndarray, loc: np.ndarray, copula: bool) -> tuple:
        d: int = S.shape[0]

        dof: float = float(theta[-1])
        if not copula:
            if loc is None:
                loc: np.ndarray = theta[:d]
            loc = loc.reshape((d, 1))

            # calculating implied shape parameter
            shape: np.ndarray = self.__cov_to_shape(S, dof)
        else:
            loc: np.ndarray = np.zeros((d, 1), dtype=float)
            shape: np.ndarray = S
        return loc, shape, dof

    def _dof_low_dim_mle(self, data: np.ndarray, **kwargs) -> tuple:
        return super()._low_dim_mle(data, **kwargs)

    def _get_low_dim_mle_objective_func_args(self, data: np.ndarray, copula: bool, cov_method: str, **kwargs) -> tuple:
        S, _, _, _ = super()._get_low_dim_mle_objective_func_args(data=data, copula=copula, cov_method=cov_method, **kwargs)
        loc: np.ndarray = data.mean(axis=0, dtype=float).flatten() if kwargs['method'] == 'dof_low_dim_mle' else None
        return S, loc, copula

    def _fit_given_data_kwargs(self, method: str, data: np.ndarray, **user_kwargs) -> dict:
        kwargs: dict = super()._fit_given_data_kwargs('low_dim_mle', data, **user_kwargs)
        if method == 'dof_low_dim_mle':
            kwargs['theta0'] = kwargs['theta0'][-1]
            kwargs['bounds'] = kwargs['bounds'][-1],
        kwargs['method'] = method
        kwargs['tol'] = 0.01
        return kwargs

    def _fit_given_params_tuple(self, params: tuple, **kwargs) -> Tuple[dict, int]:
        self._check_params(params, **kwargs)
        return {'loc': params[0], 'shape': params[1], 'dof': params[2]}, params[0].size

    def fit(self, data: dataframe_or_array = None, params: Union[Params, tuple] = None, method: str = 'dof-low-dim mle', **kwargs) -> FittedContinuousMultivariate:
        return super().fit(data=data, params=params, method=method, **kwargs)

    def num_scalar_params(self, d: int, copula: bool = False, **kwargs) -> int:
        vec_num: int = 0 if copula else d
        return 1 + vec_num + self._num_shape_scalar_params(d=d, copula=copula)


#
# if __name__ == "__main__":
#     my_mu = np.array([1, -3], dtype=float)
#     my_corr = np.array([[1, 0.7], [0.7, 1]], dtype=float)
#     my_sig = np.array([1.3, 2.5])
#     my_dof = 3.4
#
#     my_cov = np.diag(my_sig) @ my_corr @ np.diag(my_sig)
#     my_shape = my_cov * (my_dof - 2) / my_dof
#
#     rvs = multivariate_student_t.rvs(1000, (my_mu, my_shape, my_dof))
#
#     import pandas as pd
#     df = pd.DataFrame(rvs, columns=['sharks', 'lizards'])
#
#     # my_mv_t = multivariate_student_t.fit(df, copula=True, show_progress=True)
#     my_mv_t =multivariate_student_t.fit(rvs, method='dof-low-dim mle')#, show_progress=True)
#     print(my_mv_t.params.to_dict)
#
#     # my_mv_t =multivariate_student_t.fit(rvs, method='dof-low-dim mle', show_progress=True)
#     # print('here')
#     # print( my_mv_t.pdf(rvs[0, :] ))
#     # print(my_mv_t.pdf(rvs[0, :]))
#     # print(my_mv_t.logpdf(df))
#     # print('bye')
#     # print(my_mv_t.params.to_dict)
#     #
#     # my_mv_t.cdf_plot(show=False)
#     # my_mv_t.mc_cdf_plot(show=False)
#     # my_mv_t.pdf_plot(show=False)
#     # import matplotlib.pyplot as plt
#     # plt.show()
#     # # print(my_mv_t.pdf(rvs))
#     # # print(my_mv_t.cdf(rvs[:5, :]))
#     #
#     # # t2 = multivariate_student_t.fit(params=my_mv_t.params)
#     # # print(t2.mc_cdf(np.array([[1, -3]])))
#     # breakpoint()


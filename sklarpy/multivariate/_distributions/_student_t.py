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
from sklarpy.misc import CorrelationMatrix

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
        dof, loc, shape = params
        definiteness, ones = kwargs.get('definiteness', 'pd'), kwargs.get('ones', False)
        self._check_loc_shape(loc, shape, definiteness, ones)

        # checking valid dof parameter
        dof_msg: str = 'dof parameter must be a positive scalar'
        if not (isinstance(dof, float) or not isinstance(dof, int)):
            raise TypeError(dof_msg)
        elif dof <= 0:
            raise ValueError(dof_msg)

    def _logpdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        return np.array([scipy.stats.multivariate_t.logpdf(x, loc=params[1].flatten(), shape=params[2], df=params[0])], dtype=float).flatten()

    def _cdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        return np.array([scipy.stats.multivariate_t.cdf(x, loc=params[1].flatten(), shape=params[2], df=params[0])], dtype=float).flatten()

    def _rvs(self, size: int, params: tuple) -> np.ndarray:
        return scipy.stats.multivariate_t.rvs(size=size, loc=params[1].flatten(), shape=params[2], df=params[0])

    def _get_bounds(self, data: np.ndarray, as_tuple: bool, **kwargs) -> Union[dict, tuple]:
        d: int = data.shape[1]
        data_bounds: np.ndarray = np.array([data.min(axis=0), data.max(axis=0)], dtype=float).T
        default_bounds: dict = {'dof': (2.01, 100.0), 'loc': data_bounds}
        return super()._get_bounds(default_bounds, d, as_tuple, **kwargs)

    def _get_params0(self, data: np.ndarray, bounds: tuple, copula: bool, **kwargs) -> tuple:
        # getting theta0
        dof0: float = np.random.uniform(*bounds[0])
        if not copula:
            loc0: np.ndarray = data.mean(axis=0, dtype=float).flatten()
            theta0: np.ndarray = np.array([dof0, *loc0.flatten()], dtype=float)
            S: np.ndarray = CorrelationMatrix(data).cov(
                method=kwargs.get('cov_method', 'pp_kendall'))
        else:
            theta0: np.ndarray = np.array([dof0], dtype=float)
            loc = None
            S: np.ndarray = CorrelationMatrix(data).corr(
                method=kwargs.get('cov_method', 'pp_kendall'))

        return self._low_dim_theta_to_params(theta=theta0, S=S, loc=loc, copula=copula)

    def _low_dim_theta_to_params(self, theta: np.ndarray, S: np.ndarray, loc: np.ndarray, copula: bool) -> tuple:
        d: int = S.shape[0]

        dof: float = float(theta[0])
        if not copula:
            if loc is None:
                loc: np.ndarray = theta[1:d+1]
            loc = loc.reshape((d, 1))

            # calculating implied shape parameter
            shape: np.ndarray = self.__cov_to_shape(S, dof)
        else:
            loc: np.ndarray = np.zeros((d, 1), dtype=float)
            shape: np.ndarray = S
        return dof, loc, shape

    def _dof_low_dim_mle(self, data: np.ndarray, **kwargs) -> tuple:
        kwargs['params0'] = kwargs['params0'][0],
        kwargs['bounds'] = kwargs['bounds'][0],
        return super()._low_dim_mle(data, **kwargs)

    def _get_low_dim_mle_objective_func_args(self, data: np.ndarray, copula: bool, cov_method: str, **kwargs) -> tuple:
        S, _, _, _ = super()._get_low_dim_mle_objective_func_args(data=data, copula=copula, cov_method=cov_method, **kwargs)
        loc: np.ndarray = data.mean(axis=0, dtype=float).flatten() if kwargs['method'] == 'dof_low_dim_mle' else None
        return S, loc, copula

    def _fit_given_data_kwargs(self, method: str, data: np.ndarray, **user_kwargs) -> dict:
        kwargs: dict = super()._fit_given_data_kwargs('low_dim_mle', data, **user_kwargs)
        kwargs['method'] = method
        kwargs['tol'] = 0.01
        return kwargs

    def _fit_given_params_tuple(self, params: tuple, **kwargs) -> Tuple[dict, int]:
        self._check_params(params, **kwargs)
        return {'dof': params[0], 'loc': params[1], 'shape': params[2]}, params[1].size

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


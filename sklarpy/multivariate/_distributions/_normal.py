# contains code for the multivariate normal distribution
import numpy as np
import scipy.stats
from typing import Tuple, Union

from sklarpy.multivariate._prefit_dists import PreFitContinuousMultivariate
from sklarpy.misc import CorrelationMatrix
from sklarpy.multivariate._fitted_dists import FittedContinuousMultivariate
from sklarpy._other import Params
from sklarpy._utils import dataframe_or_array
from sklarpy.multivariate._distributions._params import MultivariateNormalParams


__all__ = ['multivariate_normal']


class multivariate_normal_gen(PreFitContinuousMultivariate):
    _DATA_FIT_METHODS = (*PreFitContinuousMultivariate._DATA_FIT_METHODS, 'mle')

    def _check_params(self, params: tuple, **kwargs) -> None:
        # checking correct number of params passed
        super()._check_params(params)

        # checking valid location vector and shape matrix
        loc, shape = params
        definiteness, ones = kwargs.get('definiteness', 'pd'), kwargs.get('ones', False)
        self._check_loc_shape(loc, shape, definiteness, ones)

    def _logpdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        return scipy.stats.multivariate_normal.logpdf(x, mean=params[0].flatten(), cov=params[1])

    def _cdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        return scipy.stats.multivariate_normal.cdf(x, mean=params[0].flatten(), cov=params[1])

    def _rvs(self, size: int, params: tuple) -> np.ndarray:
        return scipy.stats.multivariate_normal.rvs(size=size, mean=params[0].flatten(), cov=params[1])

    def _fit_given_data_kwargs(self, method: str, data: np.ndarray, **user_kwargs) -> dict:
        return {'cov_method': 'laloux_pp_kendall', 'copula': False}

    def _get_bounds(self, data: np.ndarray, as_tuple: bool = True, **kwargs) -> Union[dict, tuple]:
        return tuple() if as_tuple else {}

    def _mle(self, data: np.ndarray, cov_method: str, copula: bool, **kwargs) -> Tuple[tuple, bool]:
        if copula:
            loc: np.ndarray = np.zeros((data.shape[1], 1), dtype=float)
            shape: np.ndarray = CorrelationMatrix(data).corr(method=cov_method, **kwargs)
        else:
            loc: np.ndarray = data.mean(axis=0, dtype=float)
            shape: np.ndarray = CorrelationMatrix(data).cov(method=cov_method, **kwargs)
        return (loc, shape), True

    def _low_dim_theta_to_params(self, **kwargs) -> tuple:
        pass

    def _get_low_dim_theta0(self, **kwargs) -> np.ndarray:
        pass

    def _low_dim_mle(self, data: np.ndarray, cov_method: str, copula: bool, **kwargs) -> Tuple[tuple, bool]:
        return self._mle(data=data, cov_method=cov_method, copula=copula, **kwargs)

    def _fit_given_params_tuple(self, params: tuple, **kwargs) -> Tuple[dict, int]:
        self._check_params(params, **kwargs)
        return {'loc': params[0], 'shape': params[1]}, params[0].size

    def fit(self, data: dataframe_or_array = None, params: Union[Params, tuple] = None, method: str = 'mle', **kwargs) -> FittedContinuousMultivariate:
        return super().fit(data=data, params=params, method=method, **kwargs)


multivariate_normal: multivariate_normal_gen = multivariate_normal_gen(name="multivariate_normal", params_obj=MultivariateNormalParams, num_params=2, max_num_variables=np.inf)


if __name__ == "__main__":
    # my_mu = np.array([1, -3, 5], dtype=float)
    # my_corr = np.array([[1, 0.7, -0.2], [0.7, 1, -0.4], [-0.2, -0.4, 1]], dtype=float)
    # my_sig = np.array([1.3, 2.5, 1.8])

    my_mu = np.array([1, -3], dtype=float)
    my_corr = np.array([[1, 0.7], [0.7, 1]], dtype=float)
    my_sig = np.array([1.3, 2.5])

    my_cov = np.diag(my_sig) @ my_corr @ np.diag(my_sig)
    rvs = multivariate_normal.rvs(1000, (my_mu, my_cov))

    my_mv_norm = multivariate_normal.fit(rvs, copula=True)
    my_mv_norm.pdf_plot(show=False)
    my_mv_norm.cdf_plot(show=False)
    print(my_mv_norm)
    print(my_mv_norm.params.to_dict)  # need to fix name printing!
    import matplotlib.pyplot as plt
    my_mv_norm.marginal_pairplot(show=False)
    plt.show()
    # multivariate_normal.mc_cdf_plot(params=my_mv_norm.params)
    # my_mv_norm.mc_cdf_plot()

    # norm2 = multivariate_normal.fit(params=my_mv_norm.params)
    # norm2.pdf_plot()


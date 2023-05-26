# contains code for the multivariate normal distribution
import numpy as np
import scipy.stats
from typing import Tuple

from sklarpy.multivariate._prefit_dists import PreFitContinuousMultivariate
from sklarpy.multivariate._distributions._params import MultivariateNormalParams
from sklarpy.misc import CorrelationMatrix

__all__ = ['multivariate_normal']


class multivariate_normal_gen(PreFitContinuousMultivariate):
    def _pdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        return scipy.stats.multivariate_normal.pdf(x, mean=params[0].flatten(), cov=params[1])

    def _cdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        return scipy.stats.multivariate_normal.cdf(x, mean=params[0].flatten(), cov=params[1])

    def _rvs(self, size: int, params: tuple) -> np.ndarray:
        return scipy.stats.multivariate_normal.rvs(size=size, mean=params[0].flatten(), cov=params[1])

    def _logpdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        return scipy.stats.multivariate_normal.logpdf(x, mean=params[0].flatten(), cov=params[1])

    def _fit_given_data(self, data: np.ndarray, **kwargs) -> Tuple[dict, bool]:
        method: str = kwargs.pop('method', 'laloux_pp_kendall')
        mu: np.ndarray = data.mean(axis=0, dtype=float)
        cov: np.ndarray = CorrelationMatrix(data).cov(method=method, **kwargs)
        return {'mu': mu, 'shape': cov}, True

    def _fit_copula(self, data: np.ndarray, **kwargs) -> Tuple[dict, bool]:
        method: str = kwargs.pop('method', 'laloux_pp_kendall')
        mu: np.ndarray = np.zeros((data.shape[1],), dtype=float)
        corr: np.ndarray = CorrelationMatrix(data).corr(method=method, **kwargs)
        return {'mu': mu, 'shape': corr}, True

    def _fit_given_params_tuple(self, params: tuple, **kwargs) -> Tuple[dict, int]:
        # getting kwargs
        raise_cov_error: bool = kwargs.get('raise_cov_error', True)
        raise_corr_error: bool = kwargs.get('raise_corr_error', False)

        # checking correct number of parameters
        super()._fit_given_params_tuple(params)

        # checking valid mean vector and covariance matrix
        mu, cov = params
        self._check_loc_shape(mu, cov, check_shape_valid_cov=raise_cov_error, check_shape_valid_corr=raise_corr_error)
        return {'mu': mu, 'shape': cov}, mu.size


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

    my_mv_norm = multivariate_normal.fit(rvs)
    my_mv_norm.pdf_plot()
    print(my_mv_norm)
    print(my_mv_norm.params)  # need to fix name printing!
    # multivariate_normal.mc_cdf_plot(params=my_mv_norm.params)
    # my_mv_norm.mc_cdf_plot()

    norm2 = multivariate_normal.fit(params=my_mv_norm.params)
    norm2.pdf_plot()

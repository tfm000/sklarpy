import scipy.stats
import numpy as np
from typing import Tuple, Union

from sklarpy.multivariate._prefit_dists import PreFitContinuousMultivariate
from sklarpy.multivariate._fitted_dists import FittedContinuousMultivariate
from sklarpy._other import Params
from sklarpy._utils import dataframe_or_array
from sklarpy.multivariate._distributions._params import MultivariateGaussianKDEParams

__all__ = ['multivariate_gaussian_kde']


class multivariate_gaussian_kde_gen(PreFitContinuousMultivariate):
    _DATA_FIT_METHODS = 'gaussian_kde_fit',

    def _check_params(self, params: tuple, **kwargs) -> None:
        # checking correct number of params passed
        super()._check_params(params)

        # checking param is a fitted gaussian kde
        kde: scipy.stats.gaussian_kde = params[0]
        iserror: bool = not isinstance(kde, scipy.stats.gaussian_kde)
        try:
            kde.covariance
        except AttributeError:
            iserror = True
        if iserror:
            raise ValueError("kde param must be a fitted gaussian_kde object.")

    def _logpdf(self, x: np.ndarray, params: tuple,  **kwargs) -> np.ndarray:
        kde: scipy.stats.gaussian_kde = params[0]
        return kde.logpdf(x.T).T

    def _singlular_cdf(self, num_variables: int, xrow: np.ndarray, params: tuple) -> float:
        kde: scipy.stats.gaussian_kde = params[0]
        d: int = xrow.size
        ninfs: np.ndarray = np.full((d,), -np.inf)
        return kde.integrate_box(ninfs, xrow)

    def _rvs(self, size: int, params: tuple) -> np.ndarray:
        kde: scipy.stats.gaussian_kde = params[0]
        return kde.resample(size).T

    def _gaussian_kde_fit(self, data: np.ndarray, bw_method, weights, **kwargs) -> Tuple[Tuple[scipy.stats.gaussian_kde], bool]:
        return (scipy.stats.gaussian_kde(dataset=data.T, bw_method=bw_method, weights=weights),), True

    def _fit_given_data_kwargs(self, method: str, data: np.ndarray, **user_kwargs) -> dict:
        return {'bw_method': None, 'weights': None}

    def _fit_given_params_tuple(self, params: tuple, **kwargs) -> Tuple[dict, int]:
        self._check_params(params, **kwargs)
        kde: scipy.stats.gaussian_kde = params[0]
        return {'kde': kde}, kde.covariance.shape[0]

    def fit(self, data: dataframe_or_array = None, params: Union[Params, tuple] = None, **kwargs) -> FittedContinuousMultivariate:
        return super().fit(data=data, params=params, method='gaussian_kde_fit', **kwargs)


multivariate_gaussian_kde: multivariate_gaussian_kde_gen = multivariate_gaussian_kde_gen(name='multivariate_gaussian_kde', params_obj=MultivariateGaussianKDEParams, num_params=1, max_num_variables=np.inf)


if __name__ == '__main__':
    import pandas as pd
    rvs = pd.read_excel('gh_rvs.xlsx', index_col=0)
    dist = multivariate_gaussian_kde
    my_dist = dist.fit(rvs)

    print(my_dist.rvs(1000))

    import matplotlib.pyplot as plt
    # my_dist.pdf_plot(show=False)
    # my_dist.cdf_plot(show=False)
    # my_dist.mc_cdf_plot(show=False)
    my_dist.marginal_pairplot(show=False)
    plt.show()
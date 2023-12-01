# Contains code for the multivariate Gaussian KDE model
import scipy.stats
import numpy as np
import pandas as pd
from typing import Tuple, Union, Callable

from sklarpy.multivariate._prefit_dists import PreFitContinuousMultivariate
from sklarpy.multivariate._fitted_dists import FittedContinuousMultivariate
from sklarpy.utils._params import Params

__all__ = ['multivariate_gaussian_kde_gen']


class multivariate_gaussian_kde_gen(PreFitContinuousMultivariate):
    """Multivariate Gaussian KDE model."""
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

    def _get_dim(self, params: tuple) -> int:
        return params[0].covariance.shape[1]

    def _logpdf(self, x: np.ndarray, params: tuple,  **kwargs) -> np.ndarray:
        kde: scipy.stats.gaussian_kde = params[0]
        return kde.logpdf(x.T).T

    def _singlular_cdf(self, num_variables: int, xrow: np.ndarray,
                       params: tuple) -> float:
        kde: scipy.stats.gaussian_kde = params[0]
        d: int = xrow.size
        ninfs: np.ndarray = np.full((d,), -np.inf)
        return kde.integrate_box(ninfs, xrow)

    def _rvs(self, size: int, params: tuple) -> np.ndarray:
        kde: scipy.stats.gaussian_kde = params[0]
        return kde.resample(size).T

    def _gaussian_kde_fit(self, data: np.ndarray,
                          bw_method: Union[str, int, float, Callable, None],
                          weights: Union[np.ndarray, None], **kwargs) \
            -> Tuple[Tuple[scipy.stats.gaussian_kde], bool]:
        """Performs a Gaussian kernel density estimation of the data's pdf.

        Parameters
        ----------
        data: np.ndarray
            An array of multivariate data to fit the Gaussian kernel density
            too.
        bw_method : Union[str, int, float, Callable, None]
            The method used to calculate the estimator bandwidth.
            See scipy.stats.gaussian_kde for options.
        weights: Union[np.ndarray, None]
            weights of datapoints.
            See scipy.stats.gaussian_kde for options.

        See Also
        --------
        scipy.stats.gaussian_kde

        Returns
        -------
        res: Tuple[Tuple[scipy.stats.gaussian_kde], bool]
            Fitted Gaussian KDE object, True.
        """
        return (scipy.stats.gaussian_kde(dataset=data.T, bw_method=bw_method,
                                         weights=weights),), True

    def _fit_given_data_kwargs(self, method: str, data: np.ndarray,
                               **user_kwargs) -> dict:
        return {'bw_method': None, 'weights': None}

    def _fit_given_params_tuple(self, params: tuple, **kwargs) \
            -> Tuple[dict, int]:
        self._check_params(params, **kwargs)
        kde: scipy.stats.gaussian_kde = params[0]
        return {'kde': kde}, kde.covariance.shape[0]

    def fit(self, data: Union[pd.DataFrame, np.ndarray] = None,
            params: Union[Params, tuple] = None, **kwargs) \
            -> FittedContinuousMultivariate:
        """Call to fit parameters to a given dataset or to fit the
        distribution object to a set of specified parameters.

        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray]
            Optional. The multivariate dataset to fit the distribution's
            parameters too. Not required if `params` is provided.
        params : Union[Params, tuple]
            Optional. The parameters of the distribution to fit the object
            too. These can be a Params object of the specific multivariate
            distribution or a tuple containing these parameters in the correct
            order.
        kwargs:
            See below.

        Keyword arguments
        ------------------
        bw_method : Union[str, int, float, Callable, None]
            The method used to calculate the estimator bandwidth.
            See scipy.stats.gaussian_kde for options.
        weights: Union[np.ndarray, None]
            weights of datapoints.
            See scipy.stats.gaussian_kde for options.

        Returns
        --------
        fitted_multivariate: FittedContinuousMultivariate
            A fitted distribution.
        """
        kwargs.pop('method', '')
        return super().fit(data=data, params=params, method='gaussian_kde_fit',
                           **kwargs)

    def num_scalar_params(self, d: int = None, copula: bool = False, **kwargs)\
            -> int:
        return 0

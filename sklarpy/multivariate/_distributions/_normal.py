# Contains code for the multivariate normal distribution
import numpy as np
import scipy.stats
from typing import Tuple, Union

from sklarpy.multivariate._prefit_dists import PreFitContinuousMultivariate
from sklarpy.misc import CorrelationMatrix
from sklarpy.multivariate._fitted_dists import FittedContinuousMultivariate
from sklarpy._other import Params
from sklarpy._utils import dataframe_or_array

__all__ = ['multivariate_normal_gen']


class multivariate_normal_gen(PreFitContinuousMultivariate):
    """Multivariate Normal / Gaussian model."""
    _DATA_FIT_METHODS = ('mle', )

    def _check_params(self, params: tuple, **kwargs) -> None:
        # checking correct number of params passed
        super()._check_params(params)

        # checking valid location vector and shape matrix
        loc, shape = params
        definiteness, ones = (kwargs.get('definiteness', 'pd'),
                              kwargs.get('ones', False))
        self._check_loc_shape(loc, shape, definiteness, ones)

    def _logpdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        return scipy.stats.multivariate_normal.logpdf(
            x, mean=params[0].flatten(), cov=params[1])

    def _cdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        return scipy.stats.multivariate_normal.cdf(x, mean=params[0].flatten(),
                                                   cov=params[1])

    def _rvs(self, size: int, params: tuple) -> np.ndarray:
        return scipy.stats.multivariate_normal.rvs(
            size=size, mean=params[0].flatten(), cov=params[1])

    def _fit_given_data_kwargs(self, method: str, data: np.ndarray,
                               **user_kwargs) -> dict:
        return {'cov_method': 'laloux_pp_kendall', 'copula': False}

    def _get_bounds(self, data: np.ndarray, as_tuple: bool = True, **kwargs
                    ) -> Union[dict, tuple]:
        return tuple() if as_tuple else {}

    def _mle(self, data: np.ndarray, cov_method: str, copula: bool, **kwargs
             ) -> Tuple[tuple, bool]:
        """Performs MLE to fit / estimate the parameters of the distribution
        from the data.

        The MLE solution for the multivariate Normal/Gaussian distribution is
        closed form.

        See also
        --------
        CorrelationMatrix.cov

        Parameters
        -----------
        data: np.ndarray
            An array of multivariate data to optimize parameters over using
            the low dimension Maximum Likelihood Estimation (low-dim MLE)
            algorithm.
        copula: bool
            True if the distribution is a copula distribution. False otherwise.
        cov_method: str
            The method to use when estimating the sample covariance matrix.
            See CorrelationMatrix.cov for more information.
        kwargs:
            Keyword arguments to pass to CorrelationMatrix.cov

        Returns
        -------
        res: Tuple[tuple, bool]
            The parameters optimized to fit the data, True.
        """
        if copula:
            loc: np.ndarray = np.zeros((data.shape[1], 1), dtype=float)
            shape: np.ndarray = CorrelationMatrix(data).corr(method=cov_method,
                                                             **kwargs)
        else:
            loc: np.ndarray = data.mean(axis=0, dtype=float)
            shape: np.ndarray = CorrelationMatrix(data).cov(method=cov_method,
                                                            **kwargs)
        return (loc, shape), True

    def _fit_given_params_tuple(self, params: tuple, **kwargs
                                ) -> Tuple[dict, int]:
        self._check_params(params, **kwargs)
        return {'loc': params[0], 'shape': params[1]}, params[0].size

    def fit(self, data: dataframe_or_array = None,
            params: Union[Params, tuple] = None, **kwargs
            ) -> FittedContinuousMultivariate:
        """Call to fit parameters to a given dataset or to fit the
        distribution object to a set of specified parameters.

        For the multivariate Normal / Gaussian distribution, we use the
        closed-form MLE solution for all parameter fitting / optimization.

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
        cov_method: str
            When fitting to data only.
            The method to use when estimating the sample covariance matrix.
            See CorrelationMatrix.cov for more information.
            Default value is 'laloux_pp_kendall'.
        kwargs:
            kwargs for CorrelationMatrix.cov

        Returns
        --------
        fitted_multivariate: FittedContinuousMultivariate
            A fitted multivariate Normal / Gaussian distribution.
        """
        return super().fit(data=data, params=params, method='mle', **kwargs)

    def num_scalar_params(self, d: int, copula: bool = False, **kwargs) -> int:
        vec_num: int = 0 if copula else d
        return vec_num + self._num_shape_scalar_params(d=d, copula=copula)

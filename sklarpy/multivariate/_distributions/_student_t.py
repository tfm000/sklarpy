# Contains code for the multivariate Student-T distribution
import numpy as np
import pandas as pd
import scipy.stats
import scipy.integrate
import scipy.optimize
from typing import Tuple, Union

from sklarpy.multivariate._prefit_dists import PreFitContinuousMultivariate
from sklarpy.multivariate._fitted_dists import FittedContinuousMultivariate
from sklarpy.utils._params import Params
from sklarpy.misc import CorrelationMatrix

__all__ = ['multivariate_student_t_gen']


class multivariate_student_t_gen(PreFitContinuousMultivariate):
    """Multivariate Student-T model"""

    def __cov_to_shape(self, A: np.ndarray, dof: float, reverse: bool = False
                       ) -> np.ndarray:
        """Transforms a covariance matrix into a shape matrix, or vice versa.

        Parameters
        ----------
        A: np.ndarray
            The matrix to convert.
        dof: float
            The degrees of freedom parameter of the multivariate Student-T
            distribution.
        reverse: bool
            True if A is a shape matrix to convert to a covariance matrix.
            False if A is a convariance matrix to convert to a shape matrix.

        Returns
        -------
        transformed_A: np.ndarray
            shape or covariance matrix.
        """
        scale: float = (dof - 2) / dof
        if reverse:
            return A / scale if dof > 2 else A
        return A * scale if dof > 2 else A

    def _check_params(self, params: tuple, **kwargs) -> None:
        # checking correct number of params passed
        super()._check_params(params)

        # checking valid location vector and shape matrix
        dof, loc, shape = params
        definiteness, ones = (kwargs.get('definiteness', 'pd'),
                              kwargs.get('ones', False))
        self._check_loc_shape(loc, shape, definiteness, ones)

        # checking valid dof parameter
        dof_msg: str = 'dof parameter must be a positive scalar'
        if not (isinstance(dof, float) or not isinstance(dof, int)):
            raise TypeError(dof_msg)
        elif dof <= 0:
            raise ValueError(dof_msg)

    def _get_dim(self, params: tuple) -> int:
        return params[1].size

    def _logpdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        return np.array([
            scipy.stats.multivariate_t.logpdf(x, loc=params[1].flatten(),
                                              shape=params[2], df=params[0])
        ], dtype=float).flatten()

    def _cdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        return np.array([
            scipy.stats.multivariate_t.cdf(x, loc=params[1].flatten(),
                                           shape=params[2], df=params[0])
        ], dtype=float).flatten()

    def _rvs(self, size: int, params: tuple) -> np.ndarray:
        loc: np.ndarray = params[1]
        return scipy.stats.multivariate_t.rvs(
            size=size, loc=loc.flatten(), shape=params[2],
            df=params[0]).reshape((size, loc.size))

    def _get_bounds(self, data: np.ndarray, as_tuple: bool, **kwargs
                    ) -> Union[dict, tuple]:
        default_bounds: dict = {'dof': (2.01, 100.0)}
        return super()._get_bounds(default_bounds, data.shape[1], as_tuple,
                                   **kwargs)

    def _theta_to_params(self, theta: np.ndarray, mean: np.ndarray,
                         S: np.ndarray, copula: bool, **kwargs) -> tuple:
        dof: float = float(theta[0])
        if copula:
            loc: np.ndarray = np.zeros((S.shape[0], 1), dtype=float)
            shape: np.ndarray = S
        else:
            loc: np.ndarray = mean
            shape: np.ndarray = self.__cov_to_shape(S, dof)
        return dof, loc, shape

    def _params_to_theta(self, params: tuple, **kwargs) -> np.ndarray:
        return np.array([params[0]], dtype=float)

    def _get_mle_objective_func_kwargs(self, data: np.ndarray, cov_method: str,
                                       copula: bool, **kwargs) -> dict:
        # for Student-T dists, we return the args to pass to _theta_to_params
        kwargs: dict = {'copula': copula}
        d: int = data.shape[1]
        if copula:
            kwargs['mean'] = np.zeros((d, 1))
            kwargs['S'] = CorrelationMatrix(data).corr(method=cov_method,
                                                       **kwargs)
        else:
            kwargs['mean'] = data.mean(axis=0).reshape((d, 1))
            kwargs['S'] = CorrelationMatrix(data).cov(method=cov_method,
                                                      **kwargs)
        return kwargs

    def _get_params0(self, data: np.ndarray, bounds: tuple, cov_method: str,
                     copula: bool, **kwargs) -> tuple:
        # getting theta0
        dof0: float = np.random.uniform(*bounds[0])
        theta0: np.ndarray = np.array([dof0], dtype=float)

        # converting to params0
        mle_kwargs: dict = self._get_mle_objective_func_kwargs(
            data=data, cov_method=cov_method, copula=copula, **kwargs)
        return self._theta_to_params(theta0, **mle_kwargs)

    def _fit_given_params_tuple(self, params: tuple, **kwargs
                                ) -> Tuple[dict, int]:
        self._check_params(params, **kwargs)
        return ({'dof': params[0], 'loc': params[1], 'shape': params[2]},
                params[1].size)

    def num_scalar_params(self, d: int, copula: bool = False, **kwargs) -> int:
        vec_num: int = 0 if copula else d
        return 1 + vec_num + self._num_shape_scalar_params(d=d, copula=copula)

    def fit(self, data: Union[pd.DataFrame, np.ndarray] = None,
            params: Union[Params, tuple] = None,
            **kwargs) -> FittedContinuousMultivariate:
        """Call to fit parameters to a given dataset or to fit the
        distribution object to a set of specified parameters.

        For the multivariate Student-T distribution, we use Maximum-Likelihood
        Estimation to fit our parameters.

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
        bounds: tuple
            When fitting to data only.
            The bounds to use in parameter fitting / optimization, as a tuple.
            If not specified, the bounds of the degrees of freedom parameter
            are set to be (2.01, 100.0).
        params0: Union[tuple, None]
            When fitting to data only.
            An initial estimate of the parameters to use when starting the
            optimization algorithm. These can be a Params object of the
            specific multivariate distribution or a tuple containing these
            parameters in the correct order.
            For the 'em' algorithm, if params0 specified by the user, function
            outcome ~ deterministic between runs and therefore having a
            min_retries and max_retries greater than 1 has little benefit.
        kwargs:
            kwargs for CorrelationMatrix.cov

        Returns
        --------
        fitted_multivariate: FittedContinuousMultivariate
            A fitted multivariate Student-T distribution.
        """
        kwargs.pop('method', 'mle')
        return super().fit(data=data, params=params, method='mle', **kwargs)

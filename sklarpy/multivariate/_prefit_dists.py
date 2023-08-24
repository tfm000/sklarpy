from typing import Union, Callable, Tuple, Iterable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import abstractmethod
from collections import deque
import scipy.integrate
from scipy.optimize import differential_evolution

from sklarpy._other import Params
from sklarpy._utils import dataframe_or_array, TypeKeeper, check_multivariate_data, get_iterator, FitError, NotImplemented
from sklarpy._plotting import pair_plot, threeD_plot
from sklarpy.multivariate._fitted_dists import FittedContinuousMultivariate
from sklarpy.misc import CorrelationMatrix

__all__ = ['PreFitContinuousMultivariate']


class PreFitContinuousMultivariate(NotImplemented):
    _DATA_FIT_METHODS: Tuple[str] = ('low_dim_mle', )

    def __init__(self, name: str, params_obj: Params, num_params: int, max_num_variables: int):
        self._name: str = name
        self._params_obj: Params = params_obj
        self._num_params: int = num_params
        self._max_num_variables: int = max_num_variables

    def __str__(self) -> str:
        return f"PreFitContinuous{self.name.title()}Distribution"

    def __repr__(self) -> str:
        return self.__str__()

    def _get_x_array(self, x: dataframe_or_array) -> np.ndarray:
        x_array: np.ndarray = check_multivariate_data(x, allow_1d=True, allow_nans=True)

        if x_array.shape[1] > self._max_num_variables:
            raise ValueError(f"too many variables for {self.name}")

        return x_array

    def _check_loc_shape(self, loc, shape, definiteness: str, ones: bool) -> None:
        loc_error: bool = False
        if not isinstance(loc, np.ndarray):
            loc_error = True
        num_variables: int = loc.size
        if num_variables <= 0:
            loc_error = True
        if loc_error:
            raise TypeError("loc vector must be a numpy array with non-zero size.")

        if not isinstance(shape, np.ndarray):
            raise TypeError("shape matrix must be a numpy array.")
        elif (shape.shape[0] != num_variables) or (shape.shape[1] != num_variables):
            raise ValueError("shape matrix of incorrect dimension.")

        CorrelationMatrix._check_matrix('Shape', definiteness, ones, shape, True)

    @abstractmethod
    def _check_params(self, params: tuple, **kwargs) -> None:
        """raises an error is params are incorrect"""
        if len(params) != self.num_params:
            raise ValueError("Incorrect number of params given by user")

    def _get_params(self, params: Union[Params, tuple], **kwargs) -> tuple:
        if isinstance(params, self._params_obj):
            params = params.to_tuple
        elif not isinstance(params, tuple):
            raise TypeError("params must be a valid params object or a tuple.")

        if kwargs.get('check_params', True):
            self._check_params(params, **kwargs)
        return params

    def _singlular_cdf(self, num_variables: int, xrow: np.ndarray, params: tuple) -> float:
        def integrable_pdf(*xrow) -> float:
            xrow = np.asarray(xrow, dtype=float)
            return float(self.pdf(xrow, params, match_datatype=False))

        ranges = [[-np.inf, float(xrow[i])] for i in range(num_variables)]
        res: tuple = scipy.integrate.nquad(integrable_pdf, ranges)
        return res[0]

    def _cdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        num_variables: int = x.shape[1]

        show_progress: bool = kwargs.get('show_progress', True)
        iterator = get_iterator(x, show_progress, "calculating cdf values")

        return np.array([self._singlular_cdf(num_variables, xrow, params) for xrow in iterator], dtype=float)

    def _logpdf(self, x: np.ndarray, params: tuple,  **kwargs) -> np.ndarray:
        # to be overridden by child class(es)
        self._not_implemented('log-pdf')

    def _pdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        # to be overridden by child class(es)
        self._not_implemented('pdf')

    def _rvs(self, size: int, params: tuple) -> np.ndarray:
        # to be overridden by child class(es)
        self._not_implemented('rvs')

    def _logpdf_pdf_cdf(self, func_name: str, x: dataframe_or_array, params: Union[Params, tuple], match_datatype: bool = True, **kwargs) -> dataframe_or_array:
        if func_name not in ('logpdf', 'pdf', 'cdf'):
            raise ValueError("func_name invalid")

        x_array: np.ndarray = self._get_x_array(x)
        shape: tuple = x_array.shape
        output: np.ndarray = np.full((x_array.shape[0], ), np.nan)
        mask: np.ndarray = np.isnan(x_array).any(axis=1)  # only calculating for non-nan rows
        if mask.sum() == shape[0]:
            # all provided data is nans
            return output

        params_tuple: tuple = self._get_params(params, **kwargs)

        values: np.ndarray = eval(f"self._{func_name}(x_array[~mask], params_tuple, **kwargs)")
        output[~mask] = values
        return TypeKeeper(x).type_keep_from_1d_array(output, match_datatype=match_datatype, col_name=[func_name])

    def logpdf(self, x: dataframe_or_array, params: Union[Params, tuple], match_datatype: bool = True, **kwargs) -> dataframe_or_array:
        try:
            # using logpdf when possible
            return self._logpdf_pdf_cdf("logpdf", x, params, match_datatype, **kwargs)
        except NotImplementedError:
            # using pdf if possible
            try:
                pdf_values: np.ndarray = self._logpdf_pdf_cdf("pdf", x, params, False, **kwargs)
                logpdf_values: np.ndarray = np.log(pdf_values)
                return TypeKeeper(x).type_keep_from_1d_array(logpdf_values, match_datatype, col_name=['logpdf'])
            except NotImplementedError:
                # raising a function specific exception
                self._not_implemented('logpdf')

    def pdf(self, x: dataframe_or_array, params: Union[Params, tuple], match_datatype: bool = True, **kwargs) -> dataframe_or_array:
        try:
            # using logpdf when possible
            logpdf_values: Union[np.ndarray, pd.DataFrame] = self.logpdf(x, params, match_datatype)
            return np.exp(logpdf_values)
        except NotImplementedError:
            return self._logpdf_pdf_cdf("pdf", x, params, match_datatype, **kwargs)

    def cdf(self, x: dataframe_or_array, params: Union[Params, tuple], match_datatype: bool = True, **kwargs) -> dataframe_or_array:
        return self._logpdf_pdf_cdf("cdf", x, params, match_datatype, **kwargs)

    def mc_cdf(self, x: dataframe_or_array, params: Union[Params, tuple], match_datatype: bool, num_generate: int = 10 ** 4, show_progress: bool = True, **kwargs) -> dataframe_or_array:
        # converting x to a numpy array
        x_array: np.ndarray = self._get_x_array(x)
        shape: tuple = x_array.shape
        output: np.ndarray = np.full((x_array.shape[0], ), np.nan)
        mask: np.ndarray = np.isnan(x_array).any(axis=1)  # only calculating for non-nan rows
        if mask.sum() == shape[0]:
            # all provided data is nans
            return output

        # argument checks
        if not isinstance(num_generate, int) or (num_generate <= 0):
            raise TypeError("num_generate must be a positive integer")

        # whether to show progress
        iterator = get_iterator(x_array[~mask], show_progress, "calculating monte-carlo cdf values")

        # generating rvs
        rvs = kwargs.get("rvs", None)
        if rvs is None:
            rvs_array: np.ndarray = self.rvs(num_generate, params)
        else:
            # checks
            rvs_array: np.ndarray = check_multivariate_data(rvs, num_variables=x_array.shape[1])

        # calculating cdf values via mc
        mc_cdf_values: deque = deque()
        for row in iterator:
            mc_cdf_val: float = np.all(rvs_array <= row, axis=1).sum() / num_generate
            mc_cdf_values.append(mc_cdf_val)
        mc_cdf_values: np.ndarray = np.asarray(mc_cdf_values)

        output[~mask] = mc_cdf_values
        return TypeKeeper(x).type_keep_from_1d_array(output, match_datatype, col_name=['mc cdf'])

    def rvs(self, size: int, params: Union[Params, tuple]) -> np.ndarray:
        # checks
        if not isinstance(size, int):
            raise TypeError("size must be a positive integer")
        elif size <= 0:
            raise ValueError("size must be a positive integer")

        params_tuple: tuple = self._get_params(params)

        # returning rvs
        return self._rvs(size, params_tuple)

    def likelihood(self, x: dataframe_or_array, params: Union[Params, tuple]) -> float:
        try:
            loglikelihood: float = self.loglikelihood(x, params)
        except NotImplementedError:
            # raising a function specific exception
            self._not_implemented('likelihood')
        return np.exp(loglikelihood)

    def loglikelihood(self, x: dataframe_or_array, params: Union[Params, tuple], **kwargs) -> float:
        try:
            logpdf_values: np.ndarray = self.logpdf(x, params, False, **kwargs)
        except NotImplementedError:
            # raising a function specific exception
            self._not_implemented('log-likelihood')

        if np.any(np.isinf(logpdf_values)):
            # returning -np.inf instead of nan
            return -np.inf
        mask: np.ndarray = np.isnan(logpdf_values)
        return float(np.sum(logpdf_values[~mask]))

    def aic(self, data: dataframe_or_array, params: Union[Params, tuple], **kwargs) -> float:
        try:
            loglikelihood: float = self.loglikelihood(data, params)
        except NotImplementedError:
            # raising a function specific exception
            self._not_implemented('aic')
        data_array: np.ndarray = self._get_x_array(data)
        return 2 * (self.num_scalar_params(data_array.shape[1], **kwargs) - loglikelihood)

    def bic(self, data: dataframe_or_array, params: Union[Params, tuple], **kwargs) -> float:
        try:
            loglikelihood: float = self.loglikelihood(data, params)
        except NotImplementedError:
            # raising a function specific exception
            self._not_implemented('bic')

        data_array: np.ndarray = self._get_x_array(data)
        num_data_points, d = data_array.shape
        num_data_points -= np.isnan(data_array).sum()
        return -2 * loglikelihood + np.log(num_data_points) * self.num_scalar_params(d, **kwargs)

    def marginal_pairplot(self, params: Union[Params, tuple], color: str = 'royalblue', alpha: float = 1.0, figsize: tuple = (8, 8),
                          grid: bool = True, axes_names: tuple = None, plot_kde: bool = True,
                          num_generate: int = 10 ** 3, show: bool = True):

        # checking arguments
        if (not isinstance(num_generate, int)) or num_generate <= 0:
            raise TypeError("num_generate must be a posivie integer")

        rvs: np.ndarray = self.rvs(num_generate, params)  # data for plot
        plot_df: pd.DataFrame = pd.DataFrame(rvs)

        if axes_names is None:
            pass
        elif not (isinstance(axes_names, Iterable) and len(axes_names) == rvs.shape[1]):
            raise TypeError("invalid argument type in pairplot. check axes_names is None or a iterable with "
                            "an element for each variable.")

        if axes_names is not None:
            plot_df.columns = axes_names

        # plotting
        title: str = f"{self.name.replace('_', ' ').title()} Marginal Pair-Plot"
        pair_plot(plot_df, title, color, alpha, figsize, grid, plot_kde, show)

    def _pdf_cdf_mccdf_plot(self, func_str: str, var1_range: np.ndarray, var2_range: np.ndarray, params: Union[Params, tuple], color: str, alpha: float, figsize: tuple,
                            grid: bool, axes_names: Iterable, zlim: tuple, num_generate: int, num_points: int, show_progress: bool, show: bool, mc_num_generate: int = None):
        # checking arguments
        if (var1_range is not None) and (var2_range is not None):
            for var_range in (var1_range, var2_range):
                if not isinstance(var_range, np.ndarray):
                    raise TypeError("var1_range and var2_range must be None or numpy arrays.")

        else:
            rvs_array: np.ndarray = self.rvs(num_generate, params)
            if rvs_array.shape[1] != 2:
                raise NotImplementedError(f"{func_str}_plot is not implemented when the number of variables is not 2.")
            xmin, xmax = rvs_array.min(axis=0), rvs_array.max(axis=0)
            var1_range: np.ndarray = np.linspace(xmin[0], xmax[0], num_points)
            var2_range: np.ndarray = np.linspace(xmin[1], xmax[1], num_points)

        if axes_names is None:
            axes_names = ('variable 1', 'variable 2')

        if (mc_num_generate is None) and ('mc' in func_str):
            raise ValueError("mc_num_generate cannot be none for a monte-carlo function.")

        # title and name of plot to show user
        plot_name: str = func_str.replace('_', ' ').upper()
        title: str = f"{self.name.replace('_', ' ').title()} {plot_name} Plot"

        # func kwargs
        if 'mc' in func_str:
            rvs = self.rvs(mc_num_generate, params)
        else:
            rvs = None
        func_kwargs: dict = {'params': params, 'match_datatype': False, 'show_progress': False, 'rvs': rvs}
        func: Callable = eval(f"self.{func_str}")

        # plotting
        threeD_plot(func=func, var1_range=var1_range, var2_range=var2_range,
                    func_kwargs=func_kwargs, func_name=plot_name, title=title,
                    color=color, alpha=alpha, figsize=figsize, grid=grid,
                    axis_names=axes_names, zlim=zlim, show_progress=show_progress, show=show)

    def pdf_plot(self, params: Union[Params, tuple], var1_range: np.ndarray = None, var2_range: np.ndarray = None, color: str = 'royalblue', alpha: float = 1.0,
                 figsize: tuple = (8, 8), grid: bool = True, axes_names: tuple = None, zlim: tuple = (None, None),
                 num_generate: int = 1000, num_points: int = 100, show_progress: bool = True, show: bool = True) -> None:
        self._pdf_cdf_mccdf_plot(func_str='pdf', var1_range=var1_range, var2_range=var2_range, params=params, color=color, alpha=alpha, figsize=figsize, grid=grid,
                                 axes_names=axes_names, zlim=zlim, num_generate=num_generate, num_points=num_points, show_progress=show_progress, show=show)

    def cdf_plot(self, params: Union[Params, tuple], var1_range: np.ndarray = None, var2_range: np.ndarray = None, color: str = 'royalblue', alpha: float = 1.0,
                 figsize: tuple = (8, 8), grid: bool = True, axes_names: tuple = None, zlim: tuple = (0, 1),
                 num_generate: int = 1000, num_points: int = 100, show_progress: bool = True, show: bool = True) -> None:
        self._pdf_cdf_mccdf_plot(func_str='cdf', var1_range=var1_range, var2_range=var2_range, params=params, color=color, alpha=alpha, figsize=figsize, grid=grid,
                                 axes_names=axes_names, zlim=zlim, num_generate=num_generate, num_points=num_points, show_progress=show_progress, show=show)

    def mc_cdf_plot(self, params: Union[Params, tuple], var1_range: np.ndarray = None, var2_range: np.ndarray = None, mc_num_generate: int = 10 ** 4, color: str = 'royalblue', alpha: float = 1.0,
                 figsize: tuple = (8, 8), grid: bool = True, axes_names: tuple = None, zlim: tuple = (0, 1),
                 num_generate: int = 1000, num_points: int = 100, show_progress: bool = True, show: bool = True) -> None:
        self._pdf_cdf_mccdf_plot(func_str='mc_cdf', var1_range=var1_range, var2_range=var2_range, params=params, color=color, alpha=alpha, figsize=figsize, grid=grid,
                                 axes_names=axes_names, zlim=zlim, num_generate=num_generate, num_points=num_points, show_progress=show_progress, show=show, mc_num_generate=mc_num_generate)

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_params(self) -> int:
        return self._num_params

    def _num_shape_scalar_params(self, d: int, copula: bool = False) -> int:
        num_shape_scalar_params = 0.5 * d * (d-1) if copula else 0.5 * d * (d+1)
        return int(num_shape_scalar_params)

    @abstractmethod
    def num_scalar_params(self, d: int, copula: bool, **kwargs) -> int:
        pass

    @staticmethod
    def _shape_from_array(arr: np.ndarray, d: int) -> np.ndarray:
        # converts an array into shape matrix form.
        # first d values in the array are the diagonal, then the rest are the row values.
        # assumes shape matrix is symmetric
        shape: np.ndarray = np.full((d, d), np.nan, dtype=float)
        np.fill_diagonal(shape, arr[:d])
        shape_non_diag: np.ndarray = arr[d:]
        endpoint: int = 0
        for i in range(d - 1):
            startpoint = endpoint
            endpoint = int(0.5 * (i + 1) * (2 * d - i - 2))
            shape[i, i + 1:] = shape_non_diag[startpoint:endpoint]
            shape[i + 1:, i] = shape_non_diag[startpoint:endpoint]
        return shape

    # @abstractmethod
    # def _get_bounds(self, data: np.ndarray, as_tuple: bool = True, **kwargs) -> Union[dict, tuple]:
    #     bounds_dict: dict = kwargs.get('bounds', {})
    #     param_err_msg: float = "bounds must be a tuple of length 2 for scalar params or a matrix of shape (d, 2) for vector params."
    #     d: int = data.shape[1]
    #     for param, param_bounds in bounds_dict:
    #         is_error = (not (isinstance(param_bounds, tuple) or isinstance(param_bounds, np.ndarray))) or \
    #                    (isinstance(param_bounds, tuple) and len(param_bounds) != 2) or \
    #                    (isinstance(param_bounds, np.ndarray) and param_bounds.shape != (d, 2))
    #         if is_error:
    #             raise ValueError(param_err_msg)
    #     return bounds_dict

    @staticmethod
    def _bounds_dict_to_tuple(bounds_dict: dict, d: int, as_tuple: bool) -> Union[dict, tuple]:
        param_err_msg: float = "bounds must be a tuple of length 2 for scalar params or a matrix of shape (d, 2) for vector params."
        bounds_tuples: deque = deque()
        for param, param_bounds in bounds_dict.items():
            if isinstance(param_bounds, np.ndarray) and param_bounds.shape == (d, 2):
                param_bounds_tuples: tuple = tuple(map(tuple, param_bounds))
                bounds_tuples.extend(param_bounds_tuples)
            elif isinstance(param_bounds, tuple) and len(param_bounds) == 2:
                bounds_tuples.append(param_bounds)
            else:
                raise ValueError(param_err_msg)
        return tuple(bounds_tuples) if as_tuple else bounds_dict

    @abstractmethod
    def _get_bounds(self, default_bounds: dict, d: int, as_tuple: bool, **kwargs) -> Union[dict, tuple]:
        bounds_dict: dict = kwargs.get('bounds', {})
        bounds_dict = {param: bounds_dict.get(param, default_bounds[param]) for param in default_bounds}
        to_remove = ['loc'] if kwargs.get('copula', False) and 'loc' in bounds_dict else []
        return self._remove_bounds(bounds_dict, to_remove, d, as_tuple)

    def _remove_bounds(self, bounds_dict: dict, to_remove: list, d: int, as_tuple: bool) -> Union[dict, tuple]:
        for bound in to_remove:
            bounds_dict.pop(bound)
        return self._bounds_dict_to_tuple(bounds_dict, d, as_tuple)

    @abstractmethod
    def _low_dim_theta_to_params(self, theta: np.ndarray, S: np.ndarray, S_det: float, min_eig: float, copula: bool) -> tuple:
        pass

    def _low_dim_mle_objective_func(self, theta: np.ndarray, data, *args) -> float:
        params: tuple = self._low_dim_theta_to_params(theta, *args)
        return - self.loglikelihood(data, params, definiteness=None)

    @abstractmethod
    def _get_low_dim_theta0(self, data: np.ndarray, bounds: tuple, copula: bool) -> np.ndarray:
        pass

    def _get_low_dim_mle_objective_func_args(self, data: np.ndarray, copula: bool, cov_method: str, min_eig: float, **kwargs) -> tuple:
        S: np.ndarray = CorrelationMatrix(data).corr(method=cov_method) if copula else CorrelationMatrix(data).cov(method=cov_method)
        S_det: float = np.linalg.det(S)
        if min_eig is None:
            eigenvalues: np.ndarray = np.linalg.eigvals(S)
            min_eig: float = eigenvalues.min()
        return S, S_det, min_eig, copula

    def _low_dim_mle(self, data: np.ndarray, theta0: np.ndarray, copula: bool, bounds: tuple, maxiter: int, tol: float, cov_method: str, min_eig: Union[float, None], show_progress: bool, **kwargs) -> Tuple[tuple, bool]:
        # getting args to pass to optimizer
        args: tuple = self._get_low_dim_mle_objective_func_args(data, copula=copula, cov_method=cov_method, min_eig=min_eig, **kwargs)

        # running optimization
        mle_res = differential_evolution(self._low_dim_mle_objective_func, bounds, args=(data, *args), maxiter=maxiter, tol=tol, x0=theta0, disp=show_progress)
        theta: np.ndarray = mle_res['x']
        params: tuple = self._low_dim_theta_to_params(theta, *args)
        converged: bool = mle_res['success']

        if show_progress:
            print(f"Low-Dim MLE Optimisation Complete. Converged= {converged}, f(x)= {mle_res['fun']}")
        return params, converged

    @abstractmethod
    def _fit_given_data_kwargs(self, method: str, data: np.ndarray, **user_kwargs) -> dict:
        if method == 'low_dim_mle':
            bounds: tuple = self._get_bounds(data, True, **user_kwargs)
            default_theta0: np.ndarray = self._get_low_dim_theta0(data, bounds, user_kwargs.get('copula', False))
            kwargs: dict = {'theta0': default_theta0, 'copula': False,'bounds': bounds, 'maxiter': 1000, 'tol': 0.5, 'cov_method': 'pp_kendall', 'min_eig': None, 'show_progress': False}
        else:
            raise ValueError(f'{method} is not a valid method.')
        return kwargs

    def _fit_given_data(self, data: np.ndarray, method: str, **kwargs) -> Tuple[tuple, bool]:
        # getting fit method
        cleaned_method: str = method.lower().strip().replace('-', '_').replace(' ', '_')
        if cleaned_method not in self._DATA_FIT_METHODS:
            raise ValueError(f'{method} is not a valid data fitting method for {self.name}')

        # getting fit method
        data_fit_func: Callable = eval(f"self._{cleaned_method}")

        # getting additional fit args
        default_kwargs: dict = self._fit_given_data_kwargs(cleaned_method, data, **kwargs)
        kwargs_to_skip: tuple = ('q2_options', 'bounds')
        for kwarg, value in default_kwargs.items():
            if (kwarg not in kwargs) and (kwarg not in kwargs_to_skip):
                kwargs[kwarg] = value

        for kwarg in kwargs_to_skip:
            if kwarg in default_kwargs:
                kwargs[kwarg] = default_kwargs[kwarg]

        # fitting to data
        return data_fit_func(data=data, **kwargs)

    @abstractmethod
    def _fit_given_params_tuple(self, params: tuple, **kwargs) -> Tuple[dict, int]:
        if len(params) != self.num_params:
            raise ValueError("Incorrect number of params given by user")

    def fit(self, data: dataframe_or_array = None, params: Union[Params, tuple] = None, method: str = 'low-dim mle', **kwargs) -> FittedContinuousMultivariate:
        """Call to fit parameters to a given dataset or to fit the distribution object to a set of existing parameters.

        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray]
            Optional. The multivariate dataset to fit the distribution's parameters to. Not required if `params` is
            provided.
        params : Union[Params, tuple]
            Optional. The parameters of the distribution to fit the object to. Can be either a SklarPy parameter object
            (must be the correct type) or a tuple.
        method : str
            When fitting to data only.
            The method to use when fitting the distribution to the observed data.
            Default is 'low-dim mle'
        kwargs:
            See below

        Keyword arguments
        ------------------
        corr_method: str
            When fitting to data only.
            multivariate_normal and multivariate_student_t only.
            The method to use when fitting th correlation matrix to data.
            See SklarPy's CorrelationMatrix documentation for more information.
            Default is `laloux_pp_kendall`.
        bounds: dict
            When fitting to data only.
            The bounds of the parameters you are fitting.
            Must be a dictionary with parameter names as keys and values as tuples of the form
            (lower bound, upper bound) for scalar parameters or values as a (d, 2) matrix for vector parameters,
            where the left hand side is the matrix contains lower bounds and the right hand side the upper bounds.
        raise_cov_error: bool
            When fitting to user provided parameters only.
            True to raise an error if the shape matrix is an invalid covariance matrix.
            I.e. we check if the shape matrix is 2d, square, positive definite and symmetric.
            Default is True.
        print_progress: bool
            When fitting to data only.
            Available for 'low-dim mle' and 'em' algorithms.
            Prints the progress of these algorithms.
            Default is False.
        maxiter: int
            When fitting to data only.
            The maximum number of iterations an optimisation algorithm is allowed to perform.
            Default value differs depending on the optimisation algorithm / method selected.
        h: float
            When fitting to data only.
            The h parameter to use in numerical differentiation in the 'em' algorithm.
            Default value is 10 ** -5
        tol: float
            When fitting to data only.
            The tolerance to use when determing convergence.
            Default value differs depending on the optimisation algorithm / method selected.
        q2_options: dict
            When fitting to data only using the 'em' algorithm.
            Used when optimising the q2 function using scipy's differential_evolution as a part of the 'em' algorithm.
            keys must be arg / kwarg names of differerntial_evolution algorithm and values their values.
            bounds of the 'lamb', 'chi' and 'psi' parameters must not be parsed here - see the 'bounds' kwarg for this.

        Returns
        --------
        fitted_multivariate: FittedContinuousMultivariate
            A fitted distribution.
        """
        default_kwargs: dict = {'raise_cov_error': True}
        for arg in default_kwargs:
            if arg not in kwargs:
                kwargs[arg] = default_kwargs[arg]

        # TODO: fit info must contain the typekeeper for the data -> might not be possible if fitted to params
        # and likelihood, loglike, aic, bic
        # bounds/min max for all variables, stored as a np.array [[0, 1], [5, 10], [20, 30]] etc
        # cant do this directly for param fit, but, what we can do is generate a large sample of rvs and then take
        # the min and max of those as our var bounds
        fit_info: dict = {}
        if (data is None) and (params is None):
            raise ValueError("data and params cannot both be None when fitting.")
        elif params is not None:
            # User has provided a params object or tuple
            num_data_points: int = 0

            # saving params
            if isinstance(params, self._params_obj):
                params: tuple = params.to_tuple
            if isinstance(params, tuple) and len(params) == self.num_params:
                params_dict, num_variables = self._fit_given_params_tuple(params, **kwargs)
            else:
                raise TypeError(f"if params provided, must be a {self._params_obj} type or tuple of length {self.num_params}")
            params: Params = self._params_obj(params_dict, self.name, num_variables)

            # generating random data for fit evaluation stats
            data: np.ndarray = self.rvs(10**3, params)
            data_array: np.ndarray = data
            success: bool = True
        else:
            # user has provided data to fit

            # getting info from data
            data_array: np.ndarray = check_multivariate_data(data, allow_1d=True, allow_nans=False)
            num_variables: int = data_array.shape[1]
            if num_variables > self._max_num_variables:
                raise FitError(f"Too many columns in data to interpret as variables for {self.name} distribution.")

            # fitting parameters to data
            params_tuple, success = self._fit_given_data(data_array, method, **kwargs)
            params_dict, _ = self._fit_given_params_tuple(params_tuple)
            params: Params = self._params_obj(params_dict, self.name, num_variables)

            num_data_points: int = data_array.shape[0]

        # fitting TypeKeeper object
        type_keeper: TypeKeeper = TypeKeeper(data)

        # calculating fit statistics
        fit_info['likelihood'] = self.likelihood(data, params)
        fit_info['loglikelihood'] = self.loglikelihood(data, params)
        fit_info['aic'] = self.aic(data, params, **kwargs)
        fit_info['bic'] = self.bic(data, params, **kwargs)

        # calculating fit bounds
        fitted_bounds: np.ndarray = np.full((num_variables, 2), np.nan, dtype=float)
        fitted_bounds[:, 0] = data_array.min(axis=0)
        fitted_bounds[:, 1] = data_array.max(axis=0)
        fit_info['fitted_bounds'] = fitted_bounds

        # other fit values
        fit_info['type_keeper'] = type_keeper
        fit_info['params'] = params
        fit_info['num_variables'] = num_variables
        fit_info['success'] = success
        fit_info['num_data_points'] = num_data_points
        fit_info['num_params'] = len(params)
        fit_info['num_scalar_params'] = self.num_scalar_params(num_variables, **kwargs)
        return FittedContinuousMultivariate(self, fit_info)

    # TODO: have the copula classes be a separate class which inherits from each multivariate gen class -> makes param checks easier etc

    # TODO: low dim mle should not optimise for loc when copula is true!!! ugh... have 2 low dim mle funcs. one for copulas, one not

    # TODO: check what happens when 1d data parsed

    # TODO: i dont think params can be none in plots...
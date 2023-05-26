from typing import Union, Callable, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import abstractmethod
from collections import deque

from sklarpy._other import Params
from sklarpy._utils import dataframe_or_array, TypeKeeper, check_multivariate_data, get_iterator, FitError
from sklarpy._plotting import pair_plot
from sklarpy.multivariate._fitted_dists import FittedContinuousMultivariate
from sklarpy.misc import CorrelationMatrix

__all__ = ['PreFitContinuousMultivariate']


class PreFitContinuousMultivariate:
    def __init__(self, name: str, params_obj: Params, num_params: int, max_num_variables: int):
        self._name: str = name
        self._params_obj: Params = params_obj
        self._num_params: int = num_params
        self._max_num_variables: int = max_num_variables

    def __str__(self) -> str:
        return f"PreFitContinuous{self.name.title()}Distribution"

    def __repr__(self) -> str:
        return self.__str__()

    def __not_implemented(self, func_name):
        raise NotImplementedError(f"{func_name} not implemented for {self.name}")

    def _get_x_array(self, x: dataframe_or_array) -> np.ndarray:
        x_array: np.ndarray = check_multivariate_data(x)

        if not np.isnan(x_array).sum() == 0:
            raise ValueError("x cannot contain nan values.")

        if x_array.shape[1] > self._max_num_variables:
            raise ValueError(f"too many variables for {self.name}")

        return x_array

    def _get_params(self, params: Union[Params, tuple]) -> tuple:
        if isinstance(params, self._params_obj):
            params = params.to_tuple
        elif not isinstance(params, tuple):
            raise TypeError("params must be a valid params object or a tuple.")

        if len(params) != self.num_params:
            raise ValueError(f"number of params does not match that required by the model. {len(params)} != {self.num_params}")

        return params

    def _pdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        # to be overridden by child class(es)
        self.__not_implemented('pdf')

    def _cdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        # to be overridden by child class(es)
        self.__not_implemented('cdf')

    def _rvs(self, size: int, params: tuple) -> np.ndarray:
        # to be overridden by child class(es)
        self.__not_implemented('rvs')

    def _pdf_cdf(self, func_name: str, x: dataframe_or_array, params: Union[Params, tuple], match_datatype: bool = True, **kwargs) -> dataframe_or_array:
        x_array: np.ndarray = self._get_x_array(x)
        params_tuple: tuple = self._get_params(params)

        values: np.ndarray = eval(f"self._{func_name}(x_array, params_tuple, **kwargs)")

        return TypeKeeper(x).type_keep_from_1d_array(values, match_datatype, col_name=[func_name])

    def pdf(self, x: dataframe_or_array, params: Union[Params, tuple], match_datatype: bool = True, **kwargs) -> dataframe_or_array:
        return self._pdf_cdf("pdf", x, params, match_datatype, **kwargs)

    def cdf(self, x: dataframe_or_array, params: Union[Params, tuple], match_datatype: bool = True, **kwargs) -> dataframe_or_array:
        return self._pdf_cdf("cdf", x, params, match_datatype, **kwargs)

    def mc_cdf(self, x: dataframe_or_array, params: Union[Params, tuple], match_datatype: bool, rvs: np.ndarray = None, num_generate: int = 10 ** 4, show_progress: bool = True) -> dataframe_or_array:
        # converting x to a numpy array
        x_array: np.ndarray = self._get_x_array(x)

        # argument checks
        if not isinstance(num_generate, int) or (num_generate <= 0):
            raise TypeError("num_generate must be a positive integer")

        # whether to show progress
        iterator = get_iterator(x_array, show_progress, "calculating monte-carlo cdf values")

        # generating rvs
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

        return TypeKeeper(x).type_keep_from_1d_array(mc_cdf_values, match_datatype, col_name=['mc cdf'])

    def rvs(self, size: int, params: Union[Params, tuple]) -> np.ndarray:
        # checks
        if not isinstance(size, int):
            raise TypeError("size must be a positive integer")
        elif size <= 0:
            raise ValueError("size must be a positive integer")

        params_tuple: tuple = self._get_params(params)

        # returning rvs
        return self._rvs(size, params_tuple)

    def _logpdf(self, x: np.ndarray, params: tuple, **kwargs) -> np.ndarray:
        try:
            pdf_values: np.ndarray = self.pdf(x, params, False)
        except NotImplementedError:
            # raising a function specific exception
            self.__not_implemented('log-pdf')

        return np.log(pdf_values)

    def logpdf(self, x: dataframe_or_array, params: Union[Params, tuple], match_datatype: bool = True, **kwargs) -> dataframe_or_array:
        return self._pdf_cdf("logpdf", x, params, match_datatype, **kwargs)

    def likelihood(self, x: dataframe_or_array, params: Union[Params, tuple]) -> float:
        try:
            pdf_values: np.ndarray = self.pdf(x, params, False)
        except NotImplementedError:
            # raising a function specific exception
            self.__not_implemented('likelihood')

        if np.any(np.isinf(pdf_values)):
            return np.inf
        return float(np.product(pdf_values))

    def loglikelihood(self, x: dataframe_or_array, params: Union[Params, tuple]) -> float:
        try:
            logpdf_values: np.ndarray = self.logpdf(x, params, False)
        except NotImplementedError:
            # raising a function specific exception
            self.__not_implemented('log-likelihood')

        if np.any(np.isinf(logpdf_values)):
            # returning -np.inf instead of nan
            return -np.inf
        return float(np.sum(logpdf_values))

    def aic(self, x: dataframe_or_array, params: Union[Params, tuple]) -> float:
        try:
            loglikelihood: float = self.loglikelihood(x, params)
        except NotImplementedError:
            # raising a function specific exception
            self.__not_implemented('aic')
        return 2 * (self.num_params - loglikelihood)

    def bic(self, x: dataframe_or_array, params: Union[Params, tuple]) -> float:
        try:
            loglikelihood: float = self.loglikelihood(x, params)
        except NotImplementedError:
            # raising a function specific exception
            self.__not_implemented('bic')

        u_array: np.ndarray = self._get_x_array(x)
        num_data_points: int = u_array.shape[0]
        return -2 * loglikelihood + np.log(num_data_points) * self.num_params

    def marginal_pairplot(self, params: Union[Params, tuple], color: str = 'royalblue', alpha: float = 1.0, figsize: tuple = (8, 8),
                          grid: bool = True, axes_names: tuple = None, plot_kde: bool = True,
                          num_generate: int = 10 ** 3, show: bool = True):

        # checking arguments
        if axes_names is None:
            pass
        elif not (isinstance(axes_names, tuple) and len(axes_names) == self._num_variables):
            raise TypeError("invalid argument type in pairplot. check axes_names is None or a tuple with "
                            "an element for each variable.")

        if (not isinstance(num_generate, int)) or num_generate <= 0:
            raise TypeError("num_generate must be a posivie integer")

        # data for plot
        rvs: np.ndarray = self.rvs(num_generate, params)
        plot_df: pd.DataFrame = pd.DataFrame(rvs)
        if axes_names is not None:
            plot_df.columns = axes_names

        # plotting
        title: str = f'{self.name} Marginal Pair-Plot'
        pair_plot(plot_df, title, color, alpha, figsize, grid, plot_kde, show)

    def _pdf_cdf_mccdf_plot(self, func_str: str, var1_range: np.ndarray, var2_range: np.ndarray, params: Union[Params, tuple], color: str, alpha: float, figsize: tuple,
                            grid: bool, axes_names: tuple, zlim: tuple, num_generate: int, show_progress, show: bool, mc_num_generate: int = None):
        # points to plot
        if (var1_range is not None) and (var2_range is not None):
            for var_range in (var1_range, var2_range):
                if not isinstance(var_range, np.ndarray):
                    raise TypeError("var1_range and var2_range must be None or numpy arrays.")
                if var_range.ndim != 1:
                    raise ValueError("var1_range and var2_range must be 1-dimensional.")

        elif params is not None:
            rvs_array: np.ndarray = self.rvs(num_generate, params)
            if rvs_array.shape[1] != 2:
                raise NotImplementedError(f"{func_str}_plot is not implemented when the number of variables is not 2.")
            var1_range: np.ndarray = rvs_array[:, 0]
            var2_range: np.ndarray = rvs_array[:, 1]

        else:
            raise ValueError("At least one of var1_range and var2_range or params must be non-none.")

        # checking arguments
        for str_arg in (color,):
            if not isinstance(str_arg, str):
                raise TypeError(f"invalid argument in {func_str}_plot. check color is a string.")

        if not (isinstance(alpha, float) or isinstance(alpha, int)):
            raise TypeError(f"invalid argument type in {func_str}_plot. check alpha is a float or integer.")
        alpha = float(alpha)

        for tuple_arg in (figsize, zlim):
            if not (isinstance(tuple_arg, tuple) and len(tuple_arg) == 2):
                raise TypeError(f"invalid argument type in {func_str}_plot. check figsize, zlim are tuples of length 2.")

        for bool_arg in (grid, show_progress, show):
            if not isinstance(bool_arg, bool):
                raise TypeError(f"invalid argument type in {func_str}_plot. check grid, show_progress, show are boolean.")

        if axes_names is None:
            pass
        elif not (isinstance(axes_names, tuple) and len(axes_names) == 2):
            raise TypeError(f"invalid argument type in {func_str}_plot. check axes_names is None or a tuple with "
                            "an element for each variable.")

        if (mc_num_generate is None) and ('mc' in func_str):
            raise ValueError("mc_num_generate cannot be none for a monte-carlo function.")

        # name of plot to show user
        plot_name: str = func_str.replace('_', '').upper()

        # whether to show progress
        iterator = get_iterator(var2_range, show_progress, f"calculating {plot_name} values")

        # data for plot
        if 'mc' in func_str:
            rvs = self.rvs(mc_num_generate, params)
        else:
            rvs = None

        func: Callable = eval(f"self.{func_str}")
        Z: np.ndarray = np.array([[float(func(np.array([[x, y]]), params=params, match_datatype=False, show_progress=False, rvs=rvs)) for x in var1_range] for y in iterator])
        X, Y = np.meshgrid(var1_range, var2_range)

        # plotting
        fig = plt.figure(f"{self.name} {plot_name} Plot", figsize=figsize)
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, Z, color=color, alpha=alpha)
        if axes_names is not None:
            var_names = axes_names
        else:
            var_names = ['variable 1', 'variable 2']

        ax.set_xlabel(var_names[0])
        ax.set_ylabel(var_names[1])
        ax.set_zlabel(f"{plot_name.lower()} values")
        ax.set_zlim(*zlim)
        plt.title(f"{self.name} {plot_name}")
        plt.grid(grid)

        if show:
            plt.show()

    def pdf_plot(self, var1_range: np.ndarray = None, var2_range: np.ndarray = None, params: Union[Params, tuple] = None, color: str = 'royalblue', alpha: float = 1.0,
                 figsize: tuple = (8, 8), grid: bool = True, axes_names: tuple = None, zlim: tuple = (None, None),
                 num_generate: int = 100, show_progress: bool = True, show: bool = True) -> None:
        self._pdf_cdf_mccdf_plot('pdf', var1_range, var2_range, params, color, alpha, figsize, grid, axes_names, zlim, num_generate, show_progress, show)

    def cdf_plot(self, var1_range: np.ndarray = None, var2_range: np.ndarray = None, params: Union[Params, tuple] = None, color: str = 'royalblue', alpha: float = 1.0,
                 figsize: tuple = (8, 8), grid: bool = True, axes_names: tuple = None, zlim: tuple = (0, 1),
                 num_generate: int = 100, show_progress: bool = True, show: bool = True) -> None:
        self._pdf_cdf_mccdf_plot('cdf', var1_range, var2_range, params, color, alpha, figsize, grid, axes_names, zlim, num_generate, show_progress, show)

    def mc_cdf_plot(self, var1_range: np.ndarray = None, var2_range: np.ndarray = None, params: Union[Params, tuple] = None, mc_num_generate: int = 10 ** 4, color: str = 'royalblue', alpha: float = 1.0,
                 figsize: tuple = (8, 8), grid: bool = True, axes_names: tuple = None, zlim: tuple = (0, 1),
                 num_generate: int = 100, show_progress: bool = True, show: bool = True) -> None:
        self._pdf_cdf_mccdf_plot('mc_cdf', var1_range, var2_range, params, color, alpha, figsize, grid, axes_names, zlim, num_generate, show_progress, show, mc_num_generate)

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_params(self) -> int:
        return self._num_params

    @abstractmethod
    def _fit_given_data(self, data: np.ndarray, **kwargs) -> Tuple[dict, bool]:
        pass

    @abstractmethod
    def _fit_copula(self, data: np.ndarray, **kwargs) -> Tuple[dict, bool]:
        pass

    def _check_loc_shape(self, loc, shape, check_shape_valid_cov: bool = False, check_shape_valid_corr: bool = False) -> None:
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

        if check_shape_valid_cov:
            try:
                CorrelationMatrix.check_covariance_matrix(shape)
            except Exception as e:
                raise ValueError("invalid shape matrix. Must be 2d, square, positive definite and symmetric")

        if check_shape_valid_corr:
            try:
                CorrelationMatrix.check_correlation_matrix(shape)
            except Exception as e:
                raise ValueError("invalid shape matrix. Must be 2d, square, positive-semi definite, symmetric with diagonal all ones.")

    @abstractmethod
    def _fit_given_params_tuple(self, params: tuple, **kwargs) -> Tuple[dict, int]:
        if len(params) != self.num_params:
            raise ValueError("Incorrect number of params given by user")

    def fit(self, data: dataframe_or_array = None, params: Union[Params, tuple] = None, copula: bool = False, **kwargs) -> FittedContinuousMultivariate:
        """Call to fit parameters to a given dataset or to fit the distribution object to a set of existing parameters.

        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray]
            Optional. The multivariate dataset to fit the distribution's parameters to. Not required if `params` is
            provided.
        params : Union[Params, tuple]
            Optional. The parameters of the distribution to fit the object to. Can be either a SklarPy parameter object
            (must be the correct type) or a tuple.
        copula: bool
            Optional. True to fit multivariate distribution as a copula, False otherwise.
            Note that the pdf and cdf functions etc will not necessarily be that of the copula distribution
            (see SklarPy's dedicated copula distributions for those), but this method does allow for parameter
            estimation.
            Default is False.
        kwargs:
            See below

        Keyword arguments
        ------------------
        method: str
            multivariate_normal and multivariate_student_t only. The method to use when fitting the
            covariance / correlation matrix to data. See SklarPy's CorrelationMatrix documentation for more information.
            Default is `laloux_pp_kendall`.
        dof_bounds: tuple
            multivariate_student_t only. The bounds of the degrees of freedom parameter to use when fitting parameters.
            Default is `(2.01, 100.0)`.
        raise_cov_error: bool
            When fitting to user provided parameters only.
            True to raise an error if the shape matrix is an invalid covariance matrix.
            I.e. we check if the shape matrix is 2d, square, positive definite and symmetric.
            Default is True if copula is False. False otherwise.
        raise_corr_error: bool
            When fitting to user provided parameters only.
            True to raise an error if the shape matrix is an invalid correlation matrix.
            I.e. we check if the shape matrix is 2d, square, positive-semi definite, symmetric with diagonal all ones.
            Default is True if copula is True. False otherwise.
        Returns
        --------
        fitted_multivariate: FittedContinuousMultivariate
            A fitted distribution.
        """
        default_kwargs: dict = {'raise_cov_error': not copula, 'raise_corr_error': copula}
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

            # saving params
            if isinstance(params, self._params_obj):
                params: tuple = params.to_tuple
            if isinstance(params, tuple) and len(params) == self.num_params:
                params_dict, num_variables = self._fit_given_params_tuple(params, **kwargs)
            else:
                raise TypeError(f"if params provided, must be a {self._params_obj} type or tuple of length {self.num_params}")
            params: Params = self._params_obj(params_dict, self.name, num_variables)

            # for calculating fit bounds and fitting a typekeeper object
            data_array: np.ndarray = self.rvs(10**3, params)
            type_keeper: TypeKeeper = TypeKeeper(data_array)

            # calculating other fit info
            fit_info['likelihood'] = np.nan
            fit_info['loglikelihood'] = np.nan
            fit_info['aic'] = np.nan
            fit_info['bic'] = np.nan

            success: bool = True
        else:
            # user has provided data to fit

            # fitting TypeKeeper object
            type_keeper: TypeKeeper = TypeKeeper(data)

            # getting info from data
            data_array: np.ndarray = check_multivariate_data(data)
            num_variables: int = data_array.shape[1]
            if num_variables > self._max_num_variables:
                raise FitError(f"Too many columns in data to interpret as variables for {self.name} distribution.")

            # fitting parameters to data
            if copula:
                params_dict, success = self._fit_copula(data_array, **kwargs)
            else:
                params_dict, success = self._fit_given_data(data_array, **kwargs)
            params: Params = self._params_obj(params_dict, self.name, num_variables)

            # calculating other fit info
            fit_info['likelihood'] = self.likelihood(data, params)
            fit_info['loglikelihood'] = self.loglikelihood(data, params)
            fit_info['aic'] = self.aic(data, params)
            fit_info['bic'] = self.bic(data, params)

        # calculating fit bounds
        fitted_bounds: np.ndarray = np.full((num_variables, 2), np.nan, dtype=float)
        fitted_bounds[:, 0] = data_array.min(axis=0)
        fitted_bounds[:, 1] = data_array.max(axis=0)
        fit_info['fitted_bounds'] = fitted_bounds

        fit_info['type_keeper'] = type_keeper
        fit_info['params'] = params
        fit_info['num_variables'] = num_variables
        fit_info['success'] = success
        return FittedContinuousMultivariate(self, fit_info)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Tuple, Callable

from sklarpy._other import Savable, Params, Copyable
from sklarpy._utils import FitError, dataframe_or_array, none_or_array, TypeKeeper, check_multivariate_data, get_iterator
from sklarpy.copulas.marginal_fitter import MarginalFitter
from sklarpy._plotting import pair_plot

__all__ = ['Copula']


class Copula(Savable, Copyable):
    """Copula object."""
    _MAX_NUM_VARIABLES: int
    _PARAMS_OBJ: Params

    def _check_valid_u(self, u: np.ndarray):
        if not (np.all(u >= 0.0) and np.all(u <= 1.0) and (np.isnan(u).sum() == 0)):
            raise ValueError("Expected all u values to be between 0 and 1. "
                             "Check u contains valid cdf values from your marginal distributions.")

    def _check_marginals(self, marginals: Union[dataframe_or_array, MarginalFitter], **kwargs) \
            -> Tuple[none_or_array, TypeKeeper]:
        """Checks the user's provided marginals are valid and standardises them.

        Parameters
        ==========
        marginals: Union[dataframe_or_array, MarginalFitter]
            The marginals of the data to be checked. Can be either the cdf values of each random variable as a pandas
            dataframe or numpy array or a MarginalFitter object.
        kwargs:
            See below

        Keyword arguments
        =================
        univariate_fitter_options: dict
            See MarginalFitter.fit

        Returns
        ========
        check_marginals: Tuple[none_or_array, TypeKeeper]
            numpy array of marginals and a TypeKeeper object.
        """
        if marginals is None:
            return None, TypeKeeper(None)
        elif isinstance(marginals, MarginalFitter):
            if not marginals.fitted:
                # fitting MarginalFitter object to data.
                marginals.fit(**kwargs)
            marginals = marginals.marginal_cdfs(match_datatype=True)
            type_keeper: TypeKeeper = TypeKeeper(marginals)
            marginals_array: np.ndarray = check_multivariate_data(marginals)
        else:
            marginals_array: np.ndarray = check_multivariate_data(marginals)
            type_keeper: TypeKeeper = TypeKeeper(marginals)

        self._check_valid_u(marginals_array)
        return marginals_array, type_keeper

    def __init__(self, marginals: Union[dataframe_or_array, MarginalFitter] = None, name: str = None, **kwargs):
        """
        Parameters
        ==========
        marginals: Union[data_iterable, MarginalFitter]
            The marginals of random variables. Can be either the cdf values of each random variable as a pandas
            dataframe, numpy array or other data iterable or a MarginalFitter object.
            If not specified, all parameters must be specified when fitting.
        name: str
            The name of your copula object.
            This is used when saving if a file path is not specified and/or for additional identification purposes.
        kwargs:
            See below

        Keyword arguments
        =================
        univariate_fitter_options: dict
            See MarginalFitter.fit
        """
        Savable.__init__(self, name)

        checked_marginals: Tuple[none_or_array, TypeKeeper] = self._check_marginals(marginals)
        self._marginals: none_or_array = checked_marginals[0]
        self._type_keeper: TypeKeeper = checked_marginals[1]

        self._fitted: bool = False
        self._fitting: bool = False
        self._params: dict = {}
        self._num_variables: int

    def __str__(self):
        return f"{self.name}Copula(fitted={self.fitted})"

    def __repr__(self):
        return self.__str__()

    def _check_num_variables(self, num_variables):
        self._num_variables = num_variables
        if self._num_variables > self._MAX_NUM_VARIABLES:
            raise ArithmeticError(f"too many variables for {self._OBJ_NAME} in {self.name}")

    def _no_data_or_params(self):
        raise FitError(f"{self._OBJ_NAME} Object cannot be fitted without marginal cdf values or params.")

    def __not_implemented(self, func_name):
        raise NotImplementedError(f"{func_name} not implemented for {self.name}")

    def _pdf(self, u: np.ndarray, **kwargs) -> np.ndarray:
        # to be overridden by child class(es)
        self.__not_implemented('pdf')

    def _cdf(self, u: np.ndarray, **kwargs) -> np.ndarray:
        # to be overridden by child class(es)
        self.__not_implemented('cdf')

    def _rvs(self, size: int) -> np.ndarray:
        # to be overridden by child class(es)
        self.__not_implemented('rvs')

    def _fit_check(self):
        if not (self._fitting or self._fitted):
            raise FitError(f"{self.name} not fitted")

    def rvs(self, size: int, match_datatype: bool = True) -> dataframe_or_array:
        # checks
        self._fit_check()
        if not isinstance(size, int):
            raise TypeError("size must be a positive integer")
        elif size <= 0:
            raise ValueError("size must be a positive integer")

        # returning rvs
        rvs_array: np.ndarray = self._rvs(size)
        if match_datatype:
            return self._type_keeper.type_keep_from_2d_array(rvs_array)
        return rvs_array

    def _get_u_array(self, u: dataframe_or_array, generate_rvs: bool = False, rvs_size: int = 10**3) -> np.ndarray:
        self._fit_check()
        if u is None:
            marginals: none_or_array = self._marginals
            if marginals is not None:
                return marginals
            elif generate_rvs:
                return self.rvs(rvs_size)
            raise NotImplementedError("Method cannot be implemented as user did not provide marginal data when "
                                      "fitting the copula or to evaluate the method wrt. Please provide data to "
                                      "evaluate")
        if u.ndim == 1:
            u = u.reshape((1, self.num_variables))
        u = self._type_keeper.match_secondary_input(u)
        u_array: np.ndarray = np.asarray(u)
        self._check_valid_u(u_array)
        return u_array

    def _pdf_cdf(self, func_name: str, u: dataframe_or_array, match_datatype: bool, **kwargs) -> dataframe_or_array:
        if not isinstance(match_datatype, bool):
            raise TypeError("match_datatype must be a boolean.")

        u_array: np.ndarray = self._get_u_array(u)
        values: np.ndarray = eval(f"self._{func_name}(u_array, **kwargs)")
        if match_datatype:
            return self._type_keeper.type_keep_from_1d_array(values, col_name=[f"{func_name.upper()} Values"])
        return values

    def pdf(self, u: dataframe_or_array = None, match_datatype: bool = True, **kwargs) -> dataframe_or_array:
        return self._pdf_cdf("pdf", u, match_datatype, **kwargs)

    def cdf(self, u: dataframe_or_array = None, match_datatype: bool = True, **kwargs) -> dataframe_or_array:
        return self._pdf_cdf("cdf", u, match_datatype, **kwargs)

    def _params_check(self, params):
        if not isinstance(params, self._PARAMS_OBJ):
            raise TypeError(f"if params is passed, it must be a {self._PARAMS_OBJ} object.")

    def logpdf(self, u: dataframe_or_array = None, match_datatype: bool = True) -> dataframe_or_array:
        try:
            pdf_values: np.ndarray = self.pdf(u, False)
        except NotImplementedError:
            # raising a function specific exception
            self.__not_implemented('log-pdf')

        logpdf_values: np.ndarray = np.log(pdf_values)
        if match_datatype:
            return self._type_keeper.type_keep_from_1d_array(logpdf_values, col_name=["Log-PDF Values"])
        return logpdf_values

    def likelihood(self, u: dataframe_or_array = None) -> float:
        try:
            pdf_values: np.ndarray = self.pdf(u, False)
        except NotImplementedError:
            # raising a function specific exception
            self.__not_implemented('likelihood')

        if np.any(np.isinf(pdf_values)):
            return np.inf
        return float(np.product(pdf_values))

    def loglikelihood(self, u: dataframe_or_array = None) -> float:
        try:
            logpdf_values: np.ndarray = self.logpdf(u, False)
        except NotImplementedError:
            # raising a function specific exception
            self.__not_implemented('log-likelihood')

        if np.any(np.isinf(logpdf_values)):
            # returning -np.inf instead of nan
            return -np.inf
        return float(np.sum(logpdf_values))

    def aic(self, u: dataframe_or_array = None) -> float:
        try:
            loglikelihood: float = self.loglikelihood(u)
        except NotImplementedError:
            # raising a function specific exception
            self.__not_implemented('aic')
        return 2 * (self.num_params - loglikelihood)

    def bic(self, u: dataframe_or_array = None) -> float:
        try:
            loglikelihood: float = self.loglikelihood(u)
        except NotImplementedError:
            # raising a function specific exception
            self.__not_implemented('bic')

        u_array: np.ndarray = self._get_u_array(u)
        num_data_points: int = u_array.shape[0]
        return -2 * loglikelihood + np.log(num_data_points) * self.num_params

    def marginal_pairplot(self, color: str = 'royalblue', alpha: float = 1.0, figsize: tuple = (8, 8),
                          grid: bool = True, axes_names: tuple = None, plot_kde: bool = True,
                          num_generate: int = 10 ** 3, show: bool = True):
        # checks
        self._fit_check()

        # checking arguments
        if axes_names is None:
            pass
        elif not (isinstance(axes_names, tuple) and len(axes_names) == self._num_variables):
            raise TypeError("invalid argument type in pairplot. check axes_names is None or a tuple with "
                            "an element for each variable.")

        if (not isinstance(num_generate, int)) or num_generate <= 0:
            raise TypeError("num_generate must be a posivie integer")

        # data for plot
        u_array: np.ndarray = self._get_u_array(None, True, num_generate)
        plot_df: pd.DataFrame = self._type_keeper.type_keep_from_2d_array(u_array)
        if not isinstance(plot_df, pd.DataFrame):
            plot_df = pd.DataFrame(plot_df)
        if axes_names is not None:
            plot_df.columns = axes_names

        # plotting
        title: str = f'{self.name} Marginal Pair-Plot'
        pair_plot(plot_df, title, color, alpha, figsize, grid, plot_kde, show)

    def _pdf_cdf_plot(self, func_str: str, color: str, alpha: float, figsize: tuple,
                      grid: bool, axes_names: tuple, zlim: tuple, num_points: int, show_progress, show: bool):
        # checks
        self._fit_check()
        if self._num_variables != 2:
            raise NotImplementedError(f"{func_str}_plot is not implemented when the number of variables is not 2.")

        # checking arguments
        for str_arg in (color,):
            if not isinstance(str_arg, str):
                raise TypeError(f"invalid argument in {func_str}_plot. check color is a string.")

        for numeric_arg in (alpha, num_points):
            if not (isinstance(numeric_arg, float) or isinstance(numeric_arg, int)):
                raise TypeError(
                    f"invalid argument type in {func_str}_plot. check alpha, num_points is a float or integer.")
        alpha = float(alpha)
        num_points = int(num_points)

        for tuple_arg in (figsize, zlim):
            if not (isinstance(tuple_arg, tuple) and len(tuple_arg) == 2):
                raise TypeError(f"invalid argument type in {func_str}_plot. check figsize, zlim are tuples of length 2.")

        for bool_arg in (grid, show_progress, show):
            if not isinstance(bool_arg, bool):
                raise TypeError(f"invalid argument type in {func_str}_plot. check grid, show_progress, show are boolean.")

        if axes_names is None:
            pass
        elif not (isinstance(axes_names, tuple) and len(axes_names) == self._num_variables):
            raise TypeError(f"invalid argument type in {func_str}_plot. check axes_names is None or a tuple with "
                            "an element for each variable.")

        # name of plot to show user
        plot_name: str = func_str.replace('_', '').upper()

        # points to plot
        u: np.ndarray = np.linspace(0, 1, num_points)

        # whether to show progress
        iterator = get_iterator(u, show_progress, f"calculating {plot_name} values")

        # data for plot
        func: Callable = eval(f"self.{func_str}")
        Z: np.ndarray = np.array([[float(func(np.array([x, y]), match_datatype=False, show_progress=False)) for x in u] for y in iterator])
        X, Y = np.meshgrid(u, u)

        # plotting
        fig = plt.figure(f"{self.name} {plot_name} Plot", figsize=figsize)
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, Z, color=color, alpha=alpha)
        if axes_names is not None:
            var_names = axes_names
        elif self._type_keeper.original_type == pd.DataFrame:
            var_names = self._type_keeper.original_info['other']['cols']
        else:
            var_names = ['variable 1', 'variable 2']

        ax.set_xlabel(var_names[0])
        ax.set_ylabel(var_names[1])

        # u_array: np.ndarray = self._get_u_array(u, True, num_generate)
        # index: int = np.random.randint(0, u_array.shape[0], num_generate)
        # plot_df: pd.DataFrame = self._type_keeper.type_keep_from_2d_array(u_array[index, :])
        # if not isinstance(plot_df, pd.DataFrame):
        #     plot_df = pd.DataFrame(plot_df)
        # if axes_names is not None:
        #     plot_df.columns = axes_names
        # values: np.ndarray = eval(f"self.{func_str}(plot_df)")
        #
        # # plotting
        # fig = plt.figure(f"{self.name} {func_str.upper()} Plot", figsize=figsize)
        # ax = plt.axes(projection='3d')
        # ax.plot_trisurf(plot_df.iloc[:, 0], plot_df.iloc[:, 1], values, antialiased=False, linewidth=0, color=color,
        #                 alpha=alpha, vmin=0)
        # ax.set_xlabel(plot_df.columns[0])
        # ax.set_ylabel(plot_df.columns[1])
        ax.set_zlabel(f"{plot_name.lower()} values")
        ax.set_zlim(*zlim)
        plt.title(f"{self.name} {plot_name}")
        plt.grid(grid)

        if show:
            plt.show()

    def pdf_plot(self, color: str = 'royalblue', alpha: float = 1.0,
                 figsize: tuple = (8, 8), grid: bool = True, axes_names: tuple = None, zlim: tuple = (None, None),
                 num_points: int = 100, show_progress: bool = True, show: bool = True):
        self._pdf_cdf_plot('pdf', color, alpha, figsize, grid, axes_names, zlim, num_points, show_progress, show)

    def cdf_plot(self, color: str = 'royalblue', alpha: float = 1.0,
                 figsize: tuple = (8, 8), grid: bool = True, axes_names: tuple = None, zlim: tuple = (0, 1),
                 num_points: int = 100, show_progress: bool = True, show: bool = True):
        self._pdf_cdf_plot('cdf', color, alpha, figsize, grid, axes_names, zlim, num_points, show_progress, show)

    @property
    def params(self) -> Params:
        self._fit_check()
        return self._PARAMS_OBJ(self._params, f"{self._OBJ_NAME}Params")

    @property
    def num_params(self) -> int:
        return len(self.params)

    @property
    def name_with_params(self) -> str:
        """The name of the distributions with parameters (rounded to 2 significant figures if float).
        In the form: name(param0, param1, ...)
        """
        return f"{self.name}{tuple(round(p, 2) if isinstance(p, float) else p for p in self.params)}"

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def num_variables(self) -> int:
        self._fit_check()
        return self._num_variables

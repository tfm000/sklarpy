import numpy as np
import pandas as pd

from sklarpy._other import Savable, Copyable, Params
from sklarpy._utils import dataframe_or_array, TypeKeeper


class FittedContinuousMultivariate(Savable, Copyable):
    def __init__(self, obj, fit_info: dict):
        """
        Class used to hold a fitted multivariate probability distribution

        Parameters
        ----------
        obj: PreFitContinuousMultivariate
            A PreFitContinuousMultivariate object
        fit_info: dict
            A dictionary containing information about the fit.
        """
        self.__obj = obj
        self.__fit_info: dict = fit_info
        Savable.__init__(self, self.__obj.name)

    def __str__(self) -> str:
        return f"FittedContinuous{self.name.title()}Distribution"

    def __repr__(self) -> str:
        return self.__str__()

    def pdf(self, x: dataframe_or_array, match_datatype: bool = True, **kwargs) -> dataframe_or_array:
        return self.__obj.pdf(x, self.params, match_datatype, **kwargs)

    def cdf(self, x: dataframe_or_array, match_datatype: bool = True, **kwargs) -> dataframe_or_array:
        return self.__obj.cdf(x, self.params, match_datatype, **kwargs)

    def mc_cdf(self, x: dataframe_or_array, match_datatype: bool = True, num_generate: int = 10 ** 4, show_progress: bool = True, **kwargs) -> dataframe_or_array:
        return self.__obj.mc_cdf(x, self.params, match_datatype, num_generate, show_progress)

    def rvs(self, size: tuple, match_datatype: bool = True) -> dataframe_or_array:
        rvs_array: np.ndarray = self.__obj.rvs(size, self.params)
        type_keeper: TypeKeeper = self.__fit_info['type_keeper']
        return type_keeper.type_keep_from_2d_array(rvs_array, match_datatype)

    def logpdf(self, x: dataframe_or_array, match_datatype: bool = True) -> dataframe_or_array:
        return self.__obj.logpdf(x, self.params, match_datatype)

    def __likelihood_loglikelihood_aic_bic(self, func_name: str, data: dataframe_or_array = None) -> float:
        if data is None:
            return self.__fit_info[func_name]
        return eval(f"self.__obj.{func_name}(data, self.params)")

    def likelihood(self, data: dataframe_or_array = None) -> float:
        return self.__likelihood_loglikelihood_aic_bic('likelihood', data)

    def loglikelihood(self, data: dataframe_or_array = None) -> float:
        return self.__likelihood_loglikelihood_aic_bic('loglikelihood', data)

    def aic(self, data: dataframe_or_array = None) -> float:
        return self.__likelihood_loglikelihood_aic_bic('aic', data)

    def bic(self, data: dataframe_or_array = None) -> float:
        return self.__likelihood_loglikelihood_aic_bic('bic', data)

    def marginal_pairplot(self, color: str = 'royalblue', alpha: float = 1.0, figsize: tuple = (8, 8),
                          grid: bool = True, axes_names: tuple = None, plot_kde: bool = True,
                          num_generate: int = 10 ** 3, show: bool = True) -> None:
        if axes_names is None:
            type_keeper: TypeKeeper = self.__fit_info['type_keeper']
            if type_keeper.original_type == pd.DataFrame:
                axes_names = type_keeper.original_info['other']['cols']

        self.__obj.marginal_pairplot(self.params, color, alpha, figsize, grid, axes_names, plot_kde, num_generate, show)

    def __pdf_cdf_mccdf_plot(self, func_name: str, var1_range: np.ndarray, var2_range: np.ndarray, color: str,
                             alpha: float, figsize: tuple, grid: bool, axes_names: tuple, zlim: tuple, num_points: int,
                             show_progress, show: bool, mc_num_generate: int = None) -> None:
        # argument checks
        if axes_names is None:
            type_keeper: TypeKeeper = self.__fit_info['type_keeper']
            if type_keeper.original_type == pd.DataFrame:
                axes_names = type_keeper.original_info['other']['cols']

        if self.num_variables != 2:
            raise ValueError(f"{func_name}_plot is not implemented when number of variables is not 2.")

        if (not isinstance(num_points, int)) or (num_points <= 0):
            raise TypeError("num_points must be a strictly positive integer.")

        # creating our ranges
        fitted_bounds: np.ndarray = self.__fit_info['fitted_bounds']
        if var1_range is None:
            var1_range: np.ndarray = np.linspace(fitted_bounds[0][0], fitted_bounds[0][1], num_points)
        if var2_range is None:
            var2_range: np.ndarray = np.linspace(fitted_bounds[1][0], fitted_bounds[1][1], num_points)

        # plotting
        self.__obj._pdf_cdf_mccdf_plot(func_name, var1_range=var1_range, var2_range=var2_range, params=self.params,
                                       color=color, alpha=alpha, figsize=figsize, grid=grid, axes_names=axes_names,
                                       zlim=zlim, num_generate=num_points,
                                       show_progress=show_progress, show=show, mc_num_generate=mc_num_generate)

    def pdf_plot(self, var1_range: np.ndarray = None, var2_range: np.ndarray = None, color: str = 'royalblue', alpha: float = 1.0, figsize: tuple = (8, 8), grid: bool = True,
                 axes_names: tuple = None, zlim: tuple = (None, None), num_points: int = 100,
                 show_progress: bool = True, show: bool = True) -> None:
        self.__pdf_cdf_mccdf_plot('pdf', var1_range=var1_range, var2_range=var2_range, color=color, alpha=alpha, figsize=figsize, grid=grid,
                                  axes_names=axes_names, zlim=zlim, num_points=num_points,
                                  show_progress=show_progress, show=show)

    def cdf_plot(self, var1_range: np.ndarray = None, var2_range: np.ndarray = None, color: str = 'royalblue', alpha: float = 1.0, figsize: tuple = (8, 8), grid: bool = True,
                 axes_names: tuple = None, zlim: tuple = (None, None), num_points: int = 100,
                 show_progress: bool = True, show: bool = True) -> None:
        self.__pdf_cdf_mccdf_plot('cdf', var1_range=var1_range, var2_range=var2_range, color=color, alpha=alpha, figsize=figsize, grid=grid,
                                  axes_names=axes_names, zlim=zlim, num_points=num_points,
                                  show_progress=show_progress, show=show)

    def mc_cdf_plot(self, var1_range: np.ndarray = None, var2_range: np.ndarray = None, mc_num_generate: int = 10 ** 4, color: str = 'royalblue', alpha: float = 1.0,
                    figsize: tuple = (8, 8), grid: bool = True, axes_names: tuple = None,
                    zlim: tuple = (None, None), num_points: int = 100, show_progress: bool = True,
                    show: bool = True) -> None:
        self.__pdf_cdf_mccdf_plot('mc_cdf', var1_range=var1_range, var2_range=var2_range, color=color, alpha=alpha, figsize=figsize, grid=grid,
                                  axes_names=axes_names, zlim=zlim, num_points=num_points,
                                  show_progress=show_progress, show=show, mc_num_generate=mc_num_generate)
    @property
    def params(self) -> Params:
        return self.__fit_info.copy()['params']

    @property
    def num_params(self) -> int:
        return len(self.params)

    @property
    def num_variables(self) -> int:
        return self.__fit_info.copy()['num_variables']


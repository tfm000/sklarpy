import numpy as np
import pandas as pd
from typing import Union

from sklarpy._other import Savable, Copyable, Params
from sklarpy._utils import TypeKeeper

__all__ = ['FittedCopula']


class FittedCopula(Savable, Copyable):
    def __init__(self, obj, fit_info: dict):
        self.__obj = obj
        self.__fit_info = fit_info
        Savable.__init__(self, self.__obj.name)

    def __str__(self) -> str:
        return f"Fitted{self.name.title()}Copula"

    def __repr__(self) -> str:
        return self.__str__()

    def logpdf(self, x: Union[pd.DataFrame, np.ndarray], match_datatype: bool = True) -> Union[pd.DataFrame, np.ndarray]:
        return self.__obj.logpdf(x=x, copula_params=self.copula_params, mdists=self.mdists, match_datatype=match_datatype)

    def pdf(self, x: Union[pd.DataFrame, np.ndarray], match_datatype: bool = True) -> Union[pd.DataFrame, np.ndarray]:
        return self.__obj.pdf(x=x, copula_params=self.copula_params, mdists=self.mdists, match_datatype=match_datatype)

    def cdf(self, x: Union[pd.DataFrame, np.ndarray], mc_cdf: bool = False, match_datatype: bool = True, **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        return self.__obj.cdf(x=x, copula_params=self.copula_params, mdists=self.mdists, mc_cdf=mc_cdf, match_datatype=match_datatype, **kwargs)

    def rvs(self, size: int, ppf_approx: bool = True) -> np.ndarray:
        return self.__obj.rvs(size=size, copula_params=self.copula_params, mdists=self.mdists, ppf_approx=ppf_approx)

    def copula_logpdf(self, u: Union[pd.DataFrame, np.ndarray], match_datatype: bool = True) -> Union[pd.DataFrame, np.ndarray]:
        return self.__obj.copula_logpdf(u=u, copula_params=self.copula_params, match_datatype=match_datatype)

    def copula_pdf(self, u: Union[pd.DataFrame, np.ndarray], match_datatype: bool = True) -> Union[pd.DataFrame, np.ndarray]:
        return self.__obj.copula_pdf(u=u, copula_params=self.copula_params, match_datatype=match_datatype)

    def copula_cdf(self, u: Union[pd.DataFrame, np.ndarray], mc_cdf: bool = False, match_datatype: bool = True, **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        return self.__obj.copula_cdf(u=u, copula_params=self.copula_params, mc_cdf=mc_cdf, match_datatype=match_datatype, **kwargs)

    def copula_rvs(self, size: int) -> np.ndarray:
        return self.__obj.copula_rvs(size=size, copula_params=self.copula_params)

    def __likelihood_loglikelihood_aic_bic(self, func_str: str, data: Union[pd.DataFrame, np.ndarray] = None) -> float:
        if data is None:
            return self.__fit_info[func_str]
        return eval(f"self.__obj.{func_str}(data=data, copula_params=self.copula_params, mdists=self.mdists)")

    def loglikelihood(self, data: Union[np.ndarray, pd.DataFrame] = None) -> float:
        return self.__likelihood_loglikelihood_aic_bic(func_str="loglikelihood", data=data)

    def likelihood(self, data: Union[np.ndarray, pd.DataFrame] = None) -> float:
        return self.__likelihood_loglikelihood_aic_bic(func_str="likelihood", data=data)

    def aic(self, data: Union[np.ndarray, pd.DataFrame] = None) -> float:
        return self.__likelihood_loglikelihood_aic_bic(func_str="aic", data=data)

    def bic(self, data: Union[np.ndarray, pd.DataFrame] = None) -> float:
        return self.__likelihood_loglikelihood_aic_bic(func_str="bic", data=data)

    def marginal_pairplot(self, ppf_approx: bool = True, color: str = 'royalblue', alpha: float = 1.0, figsize: tuple = (8, 8),
                          grid: bool = True, axes_names: tuple = None, plot_kde: bool = True,
                          num_generate: int = 10 ** 3, show: bool = True):
        if axes_names is None:
            type_keeper: TypeKeeper = self.__fit_info['type_keeper']
            if type_keeper.original_type == pd.DataFrame:
                axes_names = type_keeper.original_info['other']['cols']

        self.__obj.marginal_pairplot(copula_params=self.copula_params, mdists=self.mdists, ppf_approx=ppf_approx,
                                     color=color, alpha=alpha, figsize=figsize, grid=grid, axes_names=axes_names, plot_kde=plot_kde,
                                     num_generate=num_generate, show=show)

    def _threeD_plot(self, func_str: str, ppf_approx: bool, var1_range: np.ndarray, var2_range: np.ndarray, color: str, alpha: float,
                 figsize: tuple, grid: bool, axes_names: tuple, zlim: tuple,
                 num_generate: int, num_points: int, show_progress: bool, show: bool, mc_num_generate: int = None) -> None:

        # argument checks
        if axes_names is None:
            type_keeper: TypeKeeper = self.__fit_info['type_keeper']
            if type_keeper.original_type == pd.DataFrame:
                axes_names = type_keeper.original_info['other']['cols']

        if self.num_variables != 2:
            raise ValueError(f"{func_str}_plot is not implemented when number of variables is not 2.")

        if (not isinstance(num_points, int)) or (num_points <= 0):
            raise TypeError("num_points must be a strictly positive integer.")

        # creating our ranges
        eps: float = 10 ** -2
        rng_bounds = np.array([[eps,1-eps], [eps, 1-eps]], dtype=float) if 'copula' in func_str else self.__fit_info['fitted_bounds']
        if var1_range is None:
            var1_range: np.ndarray = np.linspace(rng_bounds[0][0], rng_bounds[0][1], num_points)
        if var2_range is None:
            var2_range: np.ndarray = np.linspace(rng_bounds[1][0], rng_bounds[1][1], num_points)

        # plotting
        self.__obj._threeD_plot(func_str=func_str, copula_params=self.copula_params, mdists=self.mdists,
                          ppf_approx=ppf_approx, var1_range=var1_range, var2_range=var2_range,
                          color=color, alpha=alpha, figsize=figsize, grid=grid, axes_names=axes_names,
                          zlim=zlim, num_generate=num_generate, num_points=num_points, show_progress=show_progress,
                          show=show, mc_num_generate=mc_num_generate)

    def pdf_plot(self, ppf_approx: bool = True, var1_range: np.ndarray = None, var2_range: np.ndarray = None,
                 color: str = 'royalblue', alpha: float = 1.0, figsize: tuple = (8, 8), grid: bool = True, axes_names: tuple = None, zlim: tuple = (None, None),
                 num_generate: int = 1000, num_points: int = 100, show_progress: bool = True, show: bool = True) -> None:
        self._threeD_plot(func_str='pdf', ppf_approx=ppf_approx, var1_range=var1_range,
                          var2_range=var2_range, color=color, alpha=alpha, figsize=figsize, grid=grid, axes_names=axes_names, zlim=zlim,
                          num_generate=num_generate, num_points=num_points, show_progress=show_progress, show=show)

    def cdf_plot(self, ppf_approx: bool = True, var1_range: np.ndarray = None, var2_range: np.ndarray = None, color: str = 'royalblue', alpha: float = 1.0,
                 figsize: tuple = (8, 8), grid: bool = True, axes_names: tuple = None, zlim: tuple = (None, None),
                 num_generate: int = 1000, num_points: int = 100, show_progress: bool = True, show: bool = True) -> None:
        self._threeD_plot(func_str='cdf', ppf_approx=ppf_approx, var1_range=var1_range,
                          var2_range=var2_range, color=color, alpha=alpha, figsize=figsize, grid=grid, axes_names=axes_names, zlim=zlim,
                          num_generate=num_generate, num_points=num_points, show_progress=show_progress, show=show)

    def mc_cdf_plot(self, ppf_approx: bool = True, var1_range: np.ndarray = None, var2_range: np.ndarray = None, mc_num_generate: int = 10 ** 4, color: str = 'royalblue', alpha: float = 1.0,
                 figsize: tuple = (8, 8), grid: bool = True, axes_names: tuple = None, zlim: tuple = (None, None),
                 num_generate: int = 1000, num_points: int = 100, show_progress: bool = True, show: bool = True) -> None:
        self._threeD_plot(func_str='mc_cdf', ppf_approx=ppf_approx, var1_range=var1_range,
                          var2_range=var2_range, color=color, alpha=alpha, figsize=figsize, grid=grid, axes_names=axes_names, zlim=zlim,
                          num_generate=num_generate, num_points=num_points, show_progress=show_progress, show=show, mc_num_generate=mc_num_generate)

    def copula_pdf_plot(self, ppf_approx: bool = True, var1_range: np.ndarray = None, var2_range: np.ndarray = None, color: str = 'royalblue', alpha: float = 1.0,
                 figsize: tuple = (8, 8), grid: bool = True, axes_names: tuple = None, zlim: tuple = (None, None),
                 num_generate: int = 1000, num_points: int = 100, show_progress: bool = True, show: bool = True) -> None:
        self._threeD_plot(func_str='copula_pdf', ppf_approx=ppf_approx, var1_range=var1_range,
                          var2_range=var2_range, color=color, alpha=alpha, figsize=figsize, grid=grid, axes_names=axes_names, zlim=zlim,
                          num_generate=num_generate, num_points=num_points, show_progress=show_progress, show=show)

    @property
    def copula_params(self) -> Params:
        return self.__fit_info.copy()['copula_params']

    @property
    def mdists(self) -> dict:
        return self.__fit_info.copy()['mdists']

    @property
    def num_variables(self) -> int:
        return self.__fit_info.copy()['num_variables']

    @property
    def fitted_num_data_points(self) -> int:
        """The number of data points used to fit the distribution."""
        return self.__fit_info['num_data_points']

    @property
    def converged(self) -> bool:
        return self.__fit_info.copy()['success']

    @property
    def summary(self) -> pd.DataFrame:
        return self.__fit_info.copy()['summary']

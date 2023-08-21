from typing import Union, Iterable, Callable
import numpy as np
import pandas as pd

from sklarpy.copulas import MarginalFitter
from sklarpy._other import Params
from sklarpy._utils import check_multivariate_data, TypeKeeper, NotImplemented
from sklarpy.multivariate._prefit_dists import PreFitContinuousMultivariate, FittedContinuousMultivariate
from sklarpy.univariate._fitted_dists import FittedUnivariateBase
from sklarpy._plotting import pair_plot, threeD_plot
from sklarpy.copulas._fitted_dists import FittedCopula

__all__ = ['PreFitCopula']


class PreFitCopula(NotImplemented):
    def __init__(self, name: str, mv_object: PreFitContinuousMultivariate):
        self._name: str = name
        self._mv_object: PreFitContinuousMultivariate = mv_object

    def __str__(self) -> str:
        return f"PreFit{self.name}Copula"

    def __repr__(self) -> str:
        return self.__str__()

    def _get_data_array(self, data: Union[pd.DataFrame, np.ndarray], is_u: bool) -> np.ndarray:
        data_array: np.ndarray = check_multivariate_data(data=data, allow_1d=True, allow_nans=False)
        if is_u:
            if not (np.all(data_array >= 0.0) and np.all(data_array <= 1.0) and (np.isnan(data_array).sum() == 0)):
                raise ValueError("Expected all u values to be between 0 and 1. "
                                 "Check u contains valid cdf values from your marginal distributions.")
        return data_array

    def _get_mdists(self, mdists: Union[MarginalFitter, dict], d: int, check: bool = True) -> dict:
        if isinstance(mdists, MarginalFitter):
            mdists = mdists.marginals

        if check:
            if isinstance(mdists, dict):
                if len(mdists) != d:
                    raise ValueError("mdists number of distributions and the number of variables are not equal.")

                for index, dist in mdists.items():
                    if not (isinstance(index, int) and issubclass(type(dist), FittedUnivariateBase)):
                        raise ValueError('If mdists is a dictionary, it must be specified with integer keys and SklarPy fitted univariate distributions as values.')
            else:
                raise TypeError("mdists must be a dictionary or a fitted MarginalFitter object")
        return mdists

    def __mdist_calcs(self, funcs: list, data: np.ndarray, mdists: Union[MarginalFitter, dict], check: bool, funcs_kwargs: dict = None) -> dict:
        if funcs_kwargs is None:
            funcs_kwargs = {}

        d: int = data.shape[1]
        mdists_dict: dict = self._get_mdists(mdists=mdists, d=d, check=check)

        res: dict = {}
        for func in funcs:
            if func not in dir(FittedUnivariateBase):
                raise NotImplementedError(f"{func} not implemented in FittedUnivariateBase.")

            func_kwargs: dict = funcs_kwargs.get(func, {})
            func_array: np.ndarray = None
            for index, dist in mdists_dict.items():
                data_i: np.ndarray = data[:, index]
                func_str: str = f"dist.{func}(data_i, **func_kwargs)"
                vals: np.ndarray = eval(func_str)

                if func_array is None:
                    n: int = vals.size if isinstance(vals, np.ndarray) else 1
                    func_array = np.full((n, d), np.nan, dtype=float)
                func_array[:, index] = vals

            res[func] = func_array
        return res

    def logpdf(self, x: Union[pd.DataFrame, np.ndarray], copula_params: Union[Params, tuple], mdists: Union[MarginalFitter, dict], match_datatype: bool = True, **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        # checking data
        x_array: np.ndarray = self._get_data_array(data=x, is_u=False)
        mdists_dict: dict = self._get_mdists(mdists, d=x_array.shape[1], check=True)

        # calculating u values
        res: dict = self.__mdist_calcs(funcs=['cdf', 'logpdf'], data=x_array, mdists=mdists_dict, check=True)

        # calculating logpdf values
        logpdf_values: np.ndarray = self.copula_logpdf(u=res['cdf'], copula_params=copula_params, match_datatype=False, **kwargs) + res['logpdf'].sum(axis=1)
        return TypeKeeper(x).type_keep_from_1d_array(array=logpdf_values, match_datatype=match_datatype, col_name=['logpdf'])

    def pdf(self, x: Union[pd.DataFrame, np.ndarray], copula_params: Union[Params, tuple], mdists: Union[MarginalFitter, dict], match_datatype: bool = True, **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        try:
            logpdf_values: np.ndarray = self.logpdf(x=x, copula_params=copula_params, mdists=mdists, match_datatype=False, **kwargs)
        except NotImplementedError:
            # raising a function specific exception
            self._not_implemented('pdf')
        pdf_values: np.ndarray = np.exp(logpdf_values)
        return TypeKeeper(x).type_keep_from_1d_array(array=pdf_values, match_datatype=match_datatype, col_name=['pdf'])

    def cdf(self, x: Union[pd.DataFrame, np.ndarray], copula_params: Union[Params, tuple], mdists: Union[MarginalFitter, dict], mc_cdf: bool = False, match_datatype: bool = True, **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        x_array: np.ndarray = self._get_data_array(data=x, is_u=False)
        res: dict = self.__mdist_calcs(funcs=['cdf'], data=x_array, mdists=mdists, check=True)
        copula_cdf_values: np.ndarray = self.copula_cdf(u=res['cdf'], copula_params=copula_params, mc_cdf=mc_cdf, match_datatype=False, **kwargs)
        mc_str: str = "mc_" if mc_cdf else ""
        return TypeKeeper(x).type_keep_from_1d_array(array=copula_cdf_values, match_datatype=match_datatype, col_name=[f'{mc_str}cdf'])

    def rvs(self, size: int, copula_params: Union[Params, tuple], mdists: Union[MarginalFitter, dict], ppf_approx: bool = True) -> np.ndarray:
        copula_rvs: np.ndarray = self.copula_rvs(size=size, copula_params=copula_params)
        func_str: str = "ppf_approx" if ppf_approx else "ppf"
        res: dict = self.__mdist_calcs(funcs=[func_str], data=copula_rvs, mdists=mdists, check=True)
        return res[func_str]

    def _h_logpdf_sum(self, g: np.ndarray, copula_params: Union[Params, tuple]) -> np.ndarray:
        # logpdf of the marginals of G
        return np.full((g.shape[0], ), 0.0, dtype=float)

    def copula_logpdf(self, u: Union[pd.DataFrame, np.ndarray], copula_params: Union[Params, tuple], match_datatype: bool = True, **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        # checking data
        u_array: np.ndarray = self._get_data_array(data=u, is_u=True)

        # calculating copula logpdf
        g: np.ndarray = self._u_to_g(u=u_array, copula_params=copula_params)
        g_logpdf: np.ndarray = self._mv_object.logpdf(x=g, params=copula_params, match_datatype=False, **kwargs)
        copula_logpdf_values: np.ndarray = g_logpdf - self._h_logpdf_sum(g=g, copula_params=copula_params)
        return TypeKeeper(u).type_keep_from_1d_array(array=copula_logpdf_values, match_datatype=match_datatype, col_name=['copula_logpdf'])

    def copula_pdf(self, u: Union[pd.DataFrame, np.ndarray], copula_params: Union[Params, tuple], match_datatype: bool = True, **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        try:
            copula_logpdf_values: np.ndarray = self.copula_logpdf(u=u, copula_params=copula_params, match_datatype=False, **kwargs)
        except NotImplementedError:
            # raising a function specific exception
            self._not_implemented('copula_pdf')
        copula_pdf_values: np.ndarray = np.exp(copula_logpdf_values)
        return TypeKeeper(u).type_keep_from_1d_array(array=copula_pdf_values, match_datatype=match_datatype, col_name=['copula_pdf'])

    def copula_cdf(self, u: Union[pd.DataFrame, np.ndarray], copula_params: Union[Params, tuple], mc_cdf: bool = False, match_datatype: bool = True, **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        u_array: np.ndarray = self._get_data_array(data=u, is_u=True)
        g: np.ndarray = self._u_to_g(u_array, copula_params)
        mc_str: str = "mc_" if mc_cdf else ""
        func_str = f"self._mv_object.{mc_str}cdf(x=g, params=copula_params, match_datatype=False, **kwargs)"
        copula_cdf_values: np.ndarray = eval(func_str)
        return TypeKeeper(u).type_keep_from_1d_array(array=copula_cdf_values, match_datatype=match_datatype, col_name=[f'{mc_str}cdf'])

    def _g_to_u(self, g: np.ndarray, copula_params: Union[Params, tuple]) -> np.ndarray:
        # g = mv rv
        # u = copula rv
        # x = overall rv
        # i.e. for gaussian copula, g = ppf(u) and therefore u = cdf(g)
        return g

    def _u_to_g(self, u: np.ndarray, copula_params: Union[Params, tuple]):
        return u

    def copula_rvs(self, size: int, copula_params: Union[Params, tuple]) -> np.ndarray:
        mv_rvs: np.ndarray = self._mv_object.rvs(size, copula_params)
        return self._g_to_u(mv_rvs, copula_params)

    def _get_components_summary(self, fitted_mv_object: FittedContinuousMultivariate, mdists: dict, typekeeper: TypeKeeper) -> pd.DataFrame:
        # getting summary of marginal dists
        summaries: list = [dist.summary for dist in mdists.values()]
        summary: pd.DataFrame = pd.concat(summaries, axis=1)
        if typekeeper.original_type == pd.DataFrame:
            index: pd.Index = summary.index
            summary = typekeeper.type_keep_from_2d_array(np.asarray(summary))
            summary.index = index
        mv_summary: pd.DataFrame = fitted_mv_object.summary
        mv_summary.columns = [self.name]
        return pd.concat([mv_summary, summary], axis=1)

    def num_marginal_params(self, mdists: Union[MarginalFitter, dict]) -> int:
        d: int = len(mdists)
        mdists_dict: dict = self._get_mdists(mdists=mdists, d=d, check=True)
        return int(sum([dist.num_params for dist in mdists_dict.values()]))

    def num_copula_params(self, copula_params: Union[Params, dict]) -> int:
        return len(copula_params)

    def num_scalar_params(self, mdists: Union[MarginalFitter, dict]) -> int:
        return self._mv_object.num_scalar_params(d=len(mdists), copula=True) + self.num_marginal_params(mdists)

    def num_params(self, mdists: Union[MarginalFitter, dict]) -> int:
        return self._mv_object.num_params + self.num_marginal_params(mdists)

    def fit(self, data: Union[pd.DataFrame, np.ndarray] = None, copula_params: Union[Params, tuple] = None, mdists: Union[MarginalFitter, dict] = None, **kwargs) -> FittedCopula:
        if (data is None) and (copula_params is None or mdists is None):
            raise ValueError("copula_params and mdist must be provided if data is not.")

        # fitting copula
        kwargs['copula'] = True
        fitted_mv_object: FittedContinuousMultivariate = self._mv_object.fit(data=data, params=copula_params, **kwargs)

        # fitting marginal distributions
        if mdists is None:
            mdists: MarginalFitter = MarginalFitter(data=data).fit(**kwargs)
        mdists_dict: dict = self._get_mdists(mdists=mdists, d=fitted_mv_object.num_variables, check=True)

        if len(mdists_dict) != fitted_mv_object.num_variables:
            raise ValueError("number of variables of for mdist and copula params do not match.")

        # generating data to use when calculating statistics
        data = self.rvs(size=10**3, copula_params=fitted_mv_object.params, mdists=mdists_dict, ppf_approx=True) if data is None else data

        # fitting TypeKeeper object
        type_keeper: TypeKeeper = TypeKeeper(data)

        # calculating fit statistics
        loglikelihood: float = self.loglikelihood(data=data, copula_params=fitted_mv_object.params, mdists=mdists_dict)
        likelihood = np.exp(loglikelihood)
        aic: float = self.aic(data=data, copula_params=fitted_mv_object.params, mdists=mdists_dict)
        bic: float = self.bic(data=data, copula_params=fitted_mv_object.params, mdists=mdists_dict)

        fit_info: dict = {}
        fit_info['likelihood'] = likelihood
        fit_info['loglikelihood'] = loglikelihood
        fit_info['aic'] = aic
        fit_info['bic'] = bic

        # building summary
        num_params: int = self.num_params(mdists=mdists)
        num_scalar_params: int = self.num_scalar_params(mdists=mdists_dict)
        index: list = ['Distribution', '#Variables', '#Params', '#Scalar Params', 'Converged', 'Likelihood', 'Log-Likelihood', 'AIC', 'BIC', '#Fitted Data Points']
        values: list = ['Joint Distribution', fitted_mv_object.num_variables, num_params, num_scalar_params, fitted_mv_object.converged, likelihood, loglikelihood, aic, bic, fitted_mv_object.fitted_num_data_points]
        summary: pd.DataFrame = pd.DataFrame(values, index=index, columns=['Joint Distribution'])
        component_summary: pd.DataFrame = self._get_components_summary(fitted_mv_object=fitted_mv_object, mdists=mdists_dict, typekeeper=type_keeper)
        summary = pd.concat([summary, component_summary], axis=1)
        fit_info['summary'] = summary

        # calculating fit bounds
        data_array: np.ndarray = check_multivariate_data(data, allow_1d=True, allow_nans=False)
        num_variables: int = data_array.shape[1]
        fitted_bounds: np.ndarray = np.full((num_variables, 2), np.nan, dtype=float)
        fitted_bounds[:, 0] = data_array.min(axis=0)
        fitted_bounds[:, 1] = data_array.max(axis=0)
        fit_info['fitted_bounds'] = fitted_bounds

        # other fit values
        fit_info['type_keeper'] = type_keeper
        fit_info['copula_params'] = fitted_mv_object.params
        fit_info['mdists'] = mdists_dict
        fit_info['num_variables'] = num_variables
        fit_info['success'] = fitted_mv_object.converged
        fit_info['num_data_points'] = fitted_mv_object.fitted_num_data_points
        fit_info['num_params'] = num_params
        fit_info['num_scalar_params'] = num_scalar_params
        return FittedCopula(self, fit_info)

    def likelihood(self, data: Union[np.ndarray, pd.DataFrame], copula_params: Union[Params, tuple], mdists: Union[MarginalFitter, dict]) -> float:
        try:
            loglikelihood: float = self.loglikelihood(data=data, copula_params=copula_params, mdists=mdists)
        except NotImplementedError:
            # raising a function specific exception
            self._not_implemented('likelihood')
        return np.exp(loglikelihood)

    def loglikelihood(self, data: Union[np.ndarray, pd.DataFrame], copula_params: Union[Params, tuple], mdists: Union[MarginalFitter, dict]) -> float:
        try:
            logpdf_values: np.ndarray = self.logpdf(x=data, copula_params=copula_params, mdists=mdists, match_datatype=False)
        except NotImplementedError:
            # raising a function specific exception
            self._not_implemented('log-likelihood')

        if np.any(np.isinf(logpdf_values)):
            # returning -np.inf instead of nan
            return -np.inf
        return float(np.sum(logpdf_values))

    def aic(self, data: Union[pd.DataFrame, np.ndarray], copula_params: Union[Params, tuple], mdists: Union[MarginalFitter, dict]) -> float:
        try:
            loglikelihood: float = self.loglikelihood(data=data, copula_params=copula_params, mdists=mdists)
        except NotImplementedError:
            # raising a function specific exception
            self._not_implemented('aic')
        return 2 * (self.num_scalar_params(mdists=mdists) - loglikelihood)

    def bic(self, data: Union[pd.DataFrame, np.ndarray], copula_params: Union[Params, tuple], mdists: Union[MarginalFitter, dict]) -> float:
        try:
            loglikelihood: float = self.loglikelihood(data=data, copula_params=copula_params, mdists=mdists)
        except NotImplementedError:
            # raising a function specific exception
            self._not_implemented('bic')
        data_array: np.ndarray = self._get_data_array(data=data, is_u=False)
        num_data_points: int = data_array.shape[0]
        return -2 * loglikelihood + np.log(num_data_points) * self.num_scalar_params(mdists=mdists)

    def marginal_pairplot(self, copula_params: Union[Params, tuple], mdists: Union[MarginalFitter, dict], ppf_approx: bool = True, color: str = 'royalblue', alpha: float = 1.0, figsize: tuple = (8, 8),
                          grid: bool = True, axes_names: tuple = None, plot_kde: bool = True,
                          num_generate: int = 10 ** 3, show: bool = True):

        # checking arguments
        if (not isinstance(num_generate, int)) or num_generate <= 0:
            raise TypeError("num_generate must be a posivie integer")

        rvs: np.ndarray = self.rvs(size=num_generate, copula_params=copula_params, mdists=mdists, ppf_approx=ppf_approx)  # data for plot
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

    def _threeD_plot(self, func_str: str, copula_params: Union[Params, tuple], mdists: Union[MarginalFitter, dict], ppf_approx: bool, var1_range: np.ndarray, var2_range: np.ndarray, color: str, alpha: float,
                 figsize: tuple, grid: bool, axes_names: tuple, zlim: tuple,
                 num_generate: int, num_points: int, show_progress: bool, show: bool, mc_num_generate: int = None) -> None:
        # checking arguments
        if (var1_range is not None) and (var2_range is not None):
            for var_range in (var1_range, var2_range):
                if not isinstance(var_range, np.ndarray):
                    raise TypeError("var1_range and var2_range must be None or numpy arrays.")

        else:
            rvs_array: np.ndarray = self.rvs(size=num_generate, copula_params=copula_params, mdists=mdists, ppf_approx=ppf_approx)
            if rvs_array.shape[1] != 2:
                raise NotImplementedError(f"{func_str}_plot is not implemented when the number of variables is not 2.")
            eps: float = 10**-2
            xmin, xmax = (rvs_array.min(axis=0), rvs_array.max(axis=0)) if 'copula' not in func_str else (eps, 1.0-eps)
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
        func_kwargs: dict = {'copula_params': copula_params, 'mdists': mdists, 'match_datatype': False, 'show_progress': False}
        if 'mc' in func_str:
            rvs = self.rvs(size=mc_num_generate, copula_params=copula_params, mdists=mdists, ppf_approx=ppf_approx)
            func_kwargs = {**func_kwargs, **{'rvs': rvs, func_str: True}}
            func_str = func_str.replace('mc_', '')
        else:
            func_kwargs['rvs'] = None
        func: Callable = eval(f"self.{func_str}")

        # plotting
        threeD_plot(func=func, var1_range=var1_range, var2_range=var2_range,
                    func_kwargs=func_kwargs, func_name=plot_name, title=title,
                    color=color, alpha=alpha, figsize=figsize, grid=grid,
                    axis_names=axes_names, zlim=zlim, show_progress=show_progress, show=show)

    def pdf_plot(self, copula_params: Union[Params, tuple], mdists: Union[MarginalFitter, dict], ppf_approx: bool = True, var1_range: np.ndarray = None, var2_range: np.ndarray = None,
                 color: str = 'royalblue', alpha: float = 1.0, figsize: tuple = (8, 8), grid: bool = True, axes_names: tuple = None, zlim: tuple = (None, None),
                 num_generate: int = 1000, num_points: int = 100, show_progress: bool = True, show: bool = True) -> None:
        self._threeD_plot(func_str='pdf', copula_params=copula_params, mdists=mdists, ppf_approx=ppf_approx, var1_range=var1_range,
                          var2_range=var2_range, color=color, alpha=alpha, figsize=figsize, grid=grid, axes_names=axes_names, zlim=zlim,
                          num_generate=num_generate, num_points=num_points, show_progress=show_progress, show=show)

    def cdf_plot(self, copula_params: Union[Params, tuple], mdists: Union[MarginalFitter, dict], ppf_approx: bool = True, var1_range: np.ndarray = None, var2_range: np.ndarray = None, color: str = 'royalblue', alpha: float = 1.0,
                 figsize: tuple = (8, 8), grid: bool = True, axes_names: tuple = None, zlim: tuple = (None, None),
                 num_generate: int = 1000, num_points: int = 100, show_progress: bool = True, show: bool = True) -> None:
        self._threeD_plot(func_str='cdf', copula_params=copula_params, mdists=mdists, ppf_approx=ppf_approx, var1_range=var1_range,
                          var2_range=var2_range, color=color, alpha=alpha, figsize=figsize, grid=grid, axes_names=axes_names, zlim=zlim,
                          num_generate=num_generate, num_points=num_points, show_progress=show_progress, show=show)

    def mc_cdf_plot(self, copula_params: Union[Params, tuple], mdists: Union[MarginalFitter, dict], ppf_approx: bool = True, var1_range: np.ndarray = None, var2_range: np.ndarray = None, mc_num_generate: int = 10 ** 4, color: str = 'royalblue', alpha: float = 1.0,
                 figsize: tuple = (8, 8), grid: bool = True, axes_names: tuple = None, zlim: tuple = (None, None),
                 num_generate: int = 1000, num_points: int = 100, show_progress: bool = True, show: bool = True) -> None:
        self._threeD_plot(func_str='mc_cdf', copula_params=copula_params, mdists=mdists, ppf_approx=ppf_approx, var1_range=var1_range,
                          var2_range=var2_range, color=color, alpha=alpha, figsize=figsize, grid=grid, axes_names=axes_names, zlim=zlim,
                          num_generate=num_generate, num_points=num_points, show_progress=show_progress, show=show, mc_num_generate=mc_num_generate)

    def copula_pdf_plot(self, copula_params: Union[Params, tuple], mdists: Union[MarginalFitter, dict], ppf_approx: bool = True, var1_range: np.ndarray = None, var2_range: np.ndarray = None, color: str = 'royalblue', alpha: float = 1.0,
                 figsize: tuple = (8, 8), grid: bool = True, axes_names: tuple = None, zlim: tuple = (None, None),
                 num_generate: int = 1000, num_points: int = 100, show_progress: bool = True, show: bool = True) -> None:
        self._threeD_plot(func_str='copula_pdf', copula_params=copula_params, mdists=mdists, ppf_approx=ppf_approx, var1_range=var1_range,
                          var2_range=var2_range, color=color, alpha=alpha, figsize=figsize, grid=grid, axes_names=axes_names, zlim=zlim,
                          num_generate=num_generate, num_points=num_points, show_progress=show_progress, show=show)

    @property
    def name(self) -> str:
        return self._name


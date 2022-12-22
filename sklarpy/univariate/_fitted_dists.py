import os
from typing import Callable
import logging

import numpy as np
import pandas as pd
import dill
import matplotlib.pyplot as plt

from sklarpy._utils import num_or_array, SaveError, prob_bounds


class FittedUnivariateBase(object):
    def __init__(self, obj, fit_info: dict):
        self.__obj = obj
        self.__fit_info: dict = fit_info

    def __str__(self) -> str:
        return self.name_with_params

    def __repr__(self) -> str:
        return self.__str__()

    def pdf(self, x: num_or_array) -> np.ndarray:
        return self.__obj.pdf(x, self.params)

    def cdf(self, x: num_or_array) -> np.ndarray:
        return self.__obj.cdf(x, self.params)

    def ppf(self, q: num_or_array) -> np.ndarray:
        return self.__obj.ppf(q, self.params)

    def rvs(self, size: tuple) -> np.ndarray:
        return self.__obj.rvs(size, self.params)

    def logpdf(self, x: num_or_array) -> np.ndarray:
        return self.__obj.logpdf(x, self.params)

    def likelihood(self, data: np.ndarray = None) -> float:
        if data is None:
            return self.__fit_info['likelihood']
        return self.__obj.likelihood(data, self.params)

    def loglikelihood(self, data: np.ndarray = None) -> float:
        if data is None:
            return self.__fit_info['loglikelihood']
        return self.__obj.loglikelihood(data, self.params)

    def aic(self, data: np.ndarray = None) -> float:
        if data is None:
            return self.__fit_info['aic']
        return self.__obj.aic(data, self.params)

    def bic(self, data: np.ndarray = None) -> float:
        if data is None:
            return self.__fit_info['bic']
        return self.__obj.bic(data, self.params)

    def sse(self, data: np.ndarray = None) -> float:
        if data is None:
            return self.__fit_info['sse']
        return self.__obj.sse(data, self.params)

    def gof(self, data: np.ndarray = None) -> pd.DataFrame:
        if data is None:
            return self.__fit_info['gof']
        return self.__obj.gof(data, self.params)

    def plot(self, xrange: np.ndarray = None, include_empirical: bool = True, color: str = 'black', empirical_color: str = 'royalblue', qqplot_yx_color: str = 'black', alpha: float = 1.0, empirical_alpha: float = 1.0,  qqplot_yx_alpha: float = 1.0, figsize: tuple = (16, 8), grid: bool = True, num_to_plot: int = 100, show: bool = True) -> None:
        # checking arguments
        for bool_arg in (include_empirical, grid, show):
            if not isinstance(bool_arg, bool):
                raise TypeError("invalid argument type in plot. check include_empirical, empirical_hist, grid are all boolean.")

        for str_arg in (color, empirical_color, qqplot_yx_color):
            if not isinstance(str_arg, str):
                raise TypeError("invalid argument type in plot. check color, empirical_color, qqplot_yx_color are all strings.")

        for float_arg in (alpha, empirical_alpha, qqplot_yx_alpha):
            if not isinstance(float_arg, float):
                raise TypeError("invalid argument type in plot. check alpha, empirical_alpha, qqplot_yx_alpha are all floats.")

        if not (isinstance(figsize, tuple) and len(figsize) == 2):
            raise TypeError("invalid argument type in plot. check figsize is a tuple of length 2.")

        if include_empirical and not self.fitted_to_data:
            include_empirical = False
            logging.warning("include_empirical is true, but distribution was not fitted on any data. Hence we have no empirical data to display.")

        # getting xrange and qrange
        if xrange is None:
            if not (isinstance(num_to_plot, int) and num_to_plot >= 1):
                raise TypeError("invalid argument type in plot. check num_to_plot is a natural number.")

            if len(self.fitted_domain) == 2:
                # distribution was fitted using data.
                xmin, xmax = self.fitted_domain
            else:
                # distribution was fitted only using params.
                xmin = max(self.__obj.X_DATA_TYPE(self.ppf(prob_bounds[0])), self.support[0])
                xmax = min(self.__obj.X_DATA_TYPE(self.ppf(prob_bounds[1])), self.support[1])
            xrange = np.linspace(xmin, xmax, num_to_plot, dtype=self.__obj.X_DATA_TYPE)
        else:
            if not (isinstance(xrange, np.ndarray) and (xrange.size >= 1)):
                raise TypeError("invalid argument type in plot. check xrange is a np.ndarray and is non-empty.")
        qrange: np.ndarray = np.linspace(*prob_bounds, num_to_plot)

        # generating required values
        xlabels: tuple = ("x", "x", "P(X<=q)")
        ylabels: tuple = ("PDF", "P(X<=x)", "q")
        titles: tuple = ('PDF', 'CDF', 'Inverse CDF')
        distribution_values: tuple = ((xrange, self.pdf(xrange)), (xrange, self.cdf(xrange)), (qrange, self.ppf(qrange)))
        if include_empirical:
            xlabels = (*xlabels, "Theoretical Quantiles")
            ylabels = (*ylabels, "Empirical Quantiles")
            titles = (*titles, "QQ-Plot")
            empirical_pdf: Callable = self.__fit_info['empirical_pdf']
            empirical_cdf: Callable = self.__fit_info['empirical_cdf']
            empirical_ppf: Callable = self.__fit_info['empirical_ppf']
            empirical_distribution_values: tuple = ((xrange, empirical_pdf(xrange)), (xrange, empirical_cdf(xrange)), (qrange, empirical_ppf(qrange)))

        # creating plots
        subplot_dims: tuple = (1, len(xlabels))
        fig, ax = plt.subplots(*subplot_dims, figsize=figsize)
        for i in range(subplot_dims[1]):
            if i < 3:
                # plotting distribution
                ax[i].plot(distribution_values[i][0], distribution_values[i][1], label=self.name_with_params, color=color, alpha=alpha)
                if include_empirical:
                    # plotting empirical distribution
                    ax[i].plot(empirical_distribution_values[i][0], empirical_distribution_values[i][1], label="Empirical", color=empirical_color, alpha=empirical_alpha)
            elif include_empirical:
                # QQ-Plot
                ax[3].plot(xrange, xrange, label='y=x', color=qqplot_yx_color, alpha=qqplot_yx_alpha)
                ax[3].scatter(distribution_values[2][1], empirical_distribution_values[2][1], label='Quartiles', color=empirical_color, alpha=empirical_alpha)
            # labelling axis
            ax[i].set_xlabel(xlabels[i])
            ax[i].set_ylabel(ylabels[i])
            ax[i].set_title(titles[i])
            ax[i].grid(grid)
            ax[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=1)

        plt.tight_layout()
        if show:
            plt.show()

    def save(self, file_path: str = None, overwrite: bool = False, fix_extension: bool = True):
        """Saves univariate distribution as a pickled file.

        Parameters
        ==========
        file_path: Union[str, None]
            The location and file name where you are saving your distribution. If None, the distribution is saved under the distribution's name in the current working
            directory. If a file is given, it must include the full file path. The .pickle extension is optional
            provided fix_extension is True.
        overwrite: bool
            True to overwrite existing files saved under the same name. False to save under a unique name.
            Default is False.
        fix_extension: bool
            Whether to replace any existing extension with the '.pickle' file extension. Default is True.

        See Also
        ---------
        sklarpy.load
        pickle
        dill
        """
        # argument checks
        if file_path is None:
            dir_path: str = os.getcwd()
            file_path = f'{dir_path}\\{self.name}.pickle'
        elif not isinstance(file_path, str):
            raise TypeError("file argument must be a string.")

        if not isinstance(fix_extension, bool):
            raise TypeError("fix_extension argument must be a boolean.")

        # Changing file extension to .pickle
        file_name, extension = os.path.splitext(file_path)
        if fix_extension:
            extension = '.pickle'

        if not overwrite:
            # Saving under a unique file name
            count: int = 0
            unique_str: str = ''
            while os.path.exists(f'{file_name}{unique_str}{extension}'):
                count += 1
                unique_str = f'({count})'
            file_name = f'{file_name} {unique_str}'

        try:
            with open(f'{file_name}{extension}', 'wb') as f:
                dill.dump(self, f)
        except Exception as e:
            raise SaveError(e)

    @property
    def name(self) -> str:
        return self.__obj.name

    @property
    def name_with_params(self) -> str:
        return f"{self.name}{tuple(round(p, 2) for p in self.params)}"

    @property
    def summary(self) -> pd.DataFrame:
        pass

    @property
    def params(self) -> tuple:
        return self.__fit_info['params']

    @property
    def support(self) -> tuple:
        return self.__fit_info['support']

    @property
    def fitted_domain(self) -> tuple:
        return self.__fit_info['fitted_domain']

    @property
    def fitted_to_data(self) -> bool:
        return self.__fit_info['fitted_to_data']


class FittedContinuousUnivariate(FittedUnivariateBase):
    pass


class FittedDiscreteUnivariate(FittedUnivariateBase):
    pass
# Contains code for optimally fitting univariate distributions to data
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import logging
from typing import Union
from collections.abc import Iterable
import matplotlib.pyplot as plt
import math

from sklarpy.univariate.distributions import *
from sklarpy.univariate.distributions import distributions_map, all_distributions
from sklarpy._utils import SignificanceError, FitError, Utils


__all__ = ['UnivariateFitter']


class UnivariateFitter:
    """Class used for determining the best univariate/marginal probability distribution for a random sample."""

    def __init__(self, data: Union[pd.DataFrame, pd.Series, Iterable]):
        """Used for determining the best univariate/marginal probability distribution for a random sample.

        Parameters
        ==========
        data: Union[pd.DataFrame, pd.Series, Iterable]
            1 dimensional random sample data to fit distributions to.
        """
        self.data = self.__data_check(data)
        self.__init()

    def __init(self) -> None:
        """Contains attributes defined later."""
        self.__fitted_dists: dict = {}
        self.__summary: pd.DataFrame = pd.DataFrame()
        self.__datatype: str = None
        self.__fit_data = None

    @staticmethod
    def __data_check(data: Union[pd.DataFrame, pd.Series, Iterable]) -> np.ndarray:
        """Performs checks on the data given by the user.

        Parameters
        ==========
        data: Union[pd.DataFrame, pd.Series, Iterable]
            1 dimensional random sample data to fit distributions to.
        """
        if isinstance(data, pd.DataFrame):
            if len(data.columns) != 1:
                raise TypeError("Data Must be univariate.")
            return data.to_numpy().flatten()
        elif isinstance(data, pd.Series):
            return data.to_numpy().flatten()
        elif isinstance(data, Iterable):
            return np.asarray(data).flatten()
        raise TypeError("Incorrect datatype for data. Must be an Iterable, pandas DataFrame or Series.")

    def __get_distributions(self, distributions: Union[Iterable, str, None], datatype: Union[str, None],
                            multimodal: bool, numerical: bool) -> set:
        """Returns a set of distributions to fit to the data.

        Parameters
        ==========
        distributions: Union[Iterable, str, None]
            The distributions specified by the user.
            If an iterable, it must contain the string names of the
            distributions the user wants to fit.
            If a string, it must be a valid category of distributions to fit.
            Distribution categories are: 'all', 'all continuous', 'all discrete', 'all common', 'all multimodal',
            'all parametric', 'all numerical', 'all continuous parametric', 'all discrete parametric',
            'all continuous numerical', 'all discrete numerical', 'common continuous', 'common discrete',
            'continuous multimodal', 'discrete multimodal'.
            If None, either 'common discrete' or 'common continuous' will be used depending on the given data.
        datatype: Union[str, None]
            The datatype of the distributions to fit if the distributions argument is not given.
            'continuous' or 'discrete'
        multimodal: bool
            Whether to include multimodal distributions when fitting if the distributions argument is not given.
        numerical: bool
            Whether to include numerical distributions when fitting if the distributions argument is not given.

        Returns
        =======
        dists: set
            A set of distributions to fit.
        """
        if not (isinstance(distributions, Iterable) or isinstance(distributions, str) or distributions is None):
            raise ValueError("distributions must be an Iterable or None type.")

        if isinstance(distributions, str):
            distributions = distributions.lower()
            if distributions not in distributions_map:
                raise ValueError("Invalid string value for distributions.")
            dists: set = set(distributions_map[distributions])

        elif isinstance(distributions, Iterable):
            dists: set = set(distributions).intersection(all_distributions)

        else:
            datatype = self.__get_datatype(datatype)
            dists: set = set(distributions_map[f"common {datatype}"])
            if multimodal:
                dists = dists.union(distributions_map[f'{datatype} multimodal'])
            if numerical:
                dists = dists.union(f'{datatype} numerical')
        return set(name.replace('-', '_') for name in dists)

    def __get_datatype(self, datatype: Union[str, None]) -> str:
        """Performs checks on the datatype arguement.

        Parameters
        ==========
        datatype: Unions[str, None]
            The datatype argument given by the user.

        Returns
        =======
        datatype: str
            The type of data.
        """
        if datatype is None:
            return self.__datatype
        elif isinstance(datatype, str):
            datatype = datatype.lower()
            if datatype not in ('discrete', 'continuous'):
                raise ValueError('datatype must be either "continuous" or "discrete"')
            return datatype
        raise TypeError("datatype must be String or None type.")

    def _fit_single_dist(self, name: str) -> Union[tuple, None]:
        """Fits a single distribution to the data.

        Parameters
        ==========
        name: str
            The name of the distribution to fit.

        Returns
        =======
        res: tuple
            fitted, summary
        """
        try:
            dist = eval(name)

            if dist.dist_type == 'discrete' and self.__datatype == 'continuous':
                logging.warning(f'Cannot fit discrete {name} distribution to continuous data.')
                return None

            fitted = dist.fit(self.__fit_data)
            summary: pd.Series = fitted.summary
            summary.name = fitted.name
            return fitted, summary
        except Exception as e:
            logging.warning(f"Unable to fit {name} distribution.\n{e}")
        return None

    def fit(self, distributions: Union[Iterable, str, None] = None, datatype: Union[str, None] = None,
            multimodal: bool = False, numerical: bool = False, timeout: int = 10, raise_error: bool = False):
        """Fits the specified probability distributions to the data.

        Parameters
        ==========
        distributions: Union[Iterable, str, None]
            The distributions specified by the user.
            If an iterable, it must contain the string names of the
            distributions the user wants to fit.
            If a string, it must be a valid category of distributions to fit.
            Distribution categories are: 'all', 'all continuous', 'all discrete', 'all common', 'all multimodal',
            'all parametric', 'all numerical', 'all continuous parametric', 'all discrete parametric',
            'all continuous numerical', 'all discrete numerical', 'common continuous', 'common discrete',
            'continuous multimodal', 'discrete multimodal'.
            If None, either 'common discrete' or 'common continuous' will be used depending on the given data.
            Default is None.
        datatype: Union[str, None]
            The datatype of the distributions to fit if the distributions argument is not given.
            'continuous' or 'discrete'. Default is None.
        multimodal: bool
            Whether to include multimodal distributions when fitting if the distributions argument is not given.
            Default is False
        numerical: bool
            Whether to include numerical distributions when fitting if the distributions argument is not given.
            Default is False.
        timeout: int
            The maximum amount of time (seconds) to fit each distribution. Default is 10.
        raise_error: bool
            Whether to raise an error if no distributions are fitted. Default is False.

        Returns
        =======
        self
        """
        self.__fit_data, _, self.__datatype = Utils.data_type(self.data)
        distributions: set = self.__get_distributions(distributions, datatype, multimodal, numerical)
        summaries: list = []

        # fitting parametric
        parametric_dists: set = distributions.intersection(distributions_map['all parametric'])
        with ProcessPoolExecutor() as executor:
            results = executor.map(self._fit_single_dist, parametric_dists, timeout=timeout)

        for res in results:
            if res is not None:
                self.__fitted_dists[res[0].name] = res[0]
                summaries.append(res[1])

        # fitting numerical - can't be done via multiprocessing due to pickling
        numerical_dists: set = distributions.intersection(distributions_map['all numerical'])
        for name in numerical_dists:
            res = self._fit_single_dist(name)
            if res is not None:
                self.__fitted_dists[res[0].name] = res[0]
                summaries.append(res[1])

        if len(summaries) == 0:
            msg: str = "Unable to fit any distribution to the data."
            if raise_error:
                raise FitError(msg)
            else:
                logging.warning(msg)
                return self
        self.__summary = pd.concat(summaries, axis=1).T
        return self

    def get_summary(self, sortby: Union[str, None] = None, significant: bool = False, pvalue: float = 0.05) -> pd.DataFrame:
        """Returns a summary of the fitted distributions.

        Parameters
        ==========
        sortby: Union[str, None]
            The metric/column to sort the summary by. None to not sort. Default is None.
        significant: bool
            True to remove distributions which fail goodness of fit significance tests. Default is False.
        pvalue: float
            The p-value to use when rejecting the null hypothesis in goodness of fit tests. Default is 0.05.

        Returns
        =======
        summary: pd.DataFrame
            The summary of fitted distributions.
        """
        summary: pd.DataFrame = self.__summary.copy()
        if significant:
            cont_mask = np.full(len(summary), False)
            disc_mask = cont_mask.copy()
            if 'Kolmogorov-Smirnov p-value' in summary.columns:
                cont_mask = (summary['Kolmogorov-Smirnov p-value'] > pvalue) & (
                            summary['Cramér-von Mises p-value'] > pvalue)
            if 'chi-square p-value' in summary.columns:
                disc_mask = (summary['chi-square p-value'] > pvalue)
            summary = summary[cont_mask | disc_mask]
        if sortby is None:
            return summary
        return summary.sort_values(by=sortby)

    def get_best(self, significant: bool = True, pvalue: float = 0.05, raise_error: bool = False):
        """Returns the fitted probability distribution which minimises the sum of squared error between the empirical
        and fitted pdfs.

        significant: bool
            True to remove distributions which fail goodness of fit significance tests. Default is True.
        pvalue: float
            The p-value to use when rejecting the null hypothesis in goodness of fit tests. Default is 0.05.
        raise_error: bool
            True to raise an error if no distribution can be returned. If False and no distribution can be returned,
            an empirical distribution is fitted and returned. Default is False.

        Returns
        ========
        best:
            best fitted distribution.
        """
        summary = self.get_summary('Sum of Squared Error', significant, pvalue)
        if len(summary) == 0:
            if raise_error:
                raise SignificanceError("No statistically significant distributions fitted.")
            logging.warning("No statistically significant distributions fitted. Empirical distribution returned.")
            if self.__datatype == 'discrete':
                return discrete_empirical.fit(self.__fit_data)
            return empirical.fit(self.__fit_data)
        best = summary.index[0]
        return self.__fitted_dists[best]

    def plot(self, which: Union[str, Iterable] = 'all', include_sample: bool = True, hist: bool = True,
             sample_color: str = 'royalblue', sample_alpha: float = 1.0, figsize: tuple = (16, 8), grid: bool = True,
             num: int = 100) -> None:
        """Plots the fitted distributions. Produces subplots of the pdf and cdf.

        Parameters
        ==========
        which: Union[str, Iterable]
            'all', 'best' or 'significant' to plot all, the best and significant fitted distributions. If an iterable,
            the distribution names which are specified in the iterable and have been fitted are plotted.
            Default is 'all'.
        include_sample: bool
            Whether to include the sample/empirical distribution in your plots. Default is True.
        hist: bool
            True to use a histogram to represent the sample/empirical distribution of the random sample. False to plot
            a line. Default is True.
        sample_color: str
            The color of the sample/empirical distribution to use when plotting. Any acceptable value for the
            matplotlib.pyplot 'color' argument can be given. Default is 'royalblue'.
        sample_alpha: float
            The alpha of the sample distribution to use when plotting. Any acceptable value for the
            matplotlib.pyplot 'alpha' argument can be given. Default is 1.0.
        figsize: tuple
            The size of the subplot figure. Default is (16, 8).
        grid: bool
            Whether to include a grid in your subplots. Default is True.
        num: int
            The number of data points to plot. Default is 100.
        """
        if len(self.__summary) == 0:
            raise FitError("No distributions fitted.")

        if isinstance(which, str):
            if which not in ('all', 'best', 'significant'):
                raise ValueError("Invalid value for 'which' argument.")
            if which == 'all':
                dists = tuple(d for d in self.__fitted_dists.values())
            elif which == 'best':
                dists = (self.get_best(),)
            else:
                index = self.get_summary(significant=True).index
                dists = tuple(self.__fitted_dists[d] for d in index)
        else:
            dists = tuple(self.__fitted_dists[d] for d in which if d in self.__fitted_dists)

        fig, ax = plt.subplots(1, 2, figsize=figsize)
        num_dists = len(dists)

        if include_sample:
            num_dists += 1
            sample_range, sample_pdf, sample_cdf = dists[0]._fit_info['histogram']
            # sample_range, sample_pdf, sample_cdf, _ = normal._empirical(self.__fit_data)
            if hist:
                sample_plot = (lambda idx, rng, val, c, a, lab: ax[idx].bar(rng, val, color=c, alpha=a, label=lab))
            else:
                sample_plot = (lambda idx, rng, val, c, a, lab: ax[idx].plot(rng, val, color=c, alpha=a, label=lab))
            sample_plot(0, sample_range, sample_pdf, sample_color, sample_alpha, 'Sample')
            sample_plot(1, sample_range, sample_cdf, sample_color, sample_alpha, 'Sample')

        xlabels = ("x", "x")
        ylabels = ("P(X=x)", "P(X<=x)")
        titles = ('PDF', 'CDF')
        for i in range(2):
            if include_sample and sample_color == 'royalblue':
                next(ax[i]._get_lines.prop_cycler)
            ax[i].set_xlabel(xlabels[i])
            ax[i].set_ylabel(ylabels[i])
            ax[i].set_title(titles[i])
            ax[i].grid(grid)

        for dist in dists:
            if dist.dist_type == 'discrete':
                xrange = range(int(dist.fitted_domain[0]), int(dist.fitted_domain[1]) + 1)
            else:
                xrange = np.linspace(dist.fitted_domain[0], dist.fitted_domain[1], num)
            ax[0].plot(xrange, dist.pdf(xrange), label=dist.name)
            ax[1].plot(xrange, dist.cdf(xrange))  # , label=label)

        if num_dists <= 10:
            ncols = num_dists
        else:
            ncols = 10
        nrows = math.ceil(num_dists / ncols)
        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=ncols, bbox_to_anchor=(0.5, 0), fancybox=True, shadow=True)
        plt.tight_layout()
        bottom = max(nrows * 0.2 / figsize[1], 0.12)
        fig.subplots_adjust(bottom=bottom)
        plt.show()

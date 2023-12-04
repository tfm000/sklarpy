# Contains a code to determine the univariate distribution which best fits a
# dataset
import numpy as np
from typing import Union, Iterable, Callable
import logging
import pandas as pd
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from pickle import PicklingError
import matplotlib.pyplot as plt
import math
import warnings

from sklarpy.univariate.distributions import *
from sklarpy.univariate.distributions_map import distributions_map
from sklarpy.utils._errors import SignificanceError, FitError
from sklarpy.utils._input_handlers import check_univariate_data, \
    check_array_datatype
from sklarpy.utils._serialize import Savable

__all__ = ['UnivariateFitter']


class UnivariateFitter(Savable):
    """Used for determining the best univariate/marginal distribution
    for a random sample."""
    _OBJ_NAME = 'UnivariateFitter'

    def __init__(self, data: Union[pd.DataFrame, pd.Series, np.ndarray,
                                   Iterable], name: str = None):
        """Used for determining the best univariate/marginal distribution for
        a random sample.

        Parameters
        ----------
        data : Union[pd.DataFrame, pd.Series, np.ndarray, Iterable]
            The data to fit univariate distributions too.
            Can be a pd.DataFrame, pd.Series, np.ndarray or any other iterable
            containing data. Data may be continuous or discrete.
        name: str
            The name of your UnivariateFitter object. Used when saving.
            If none, 'UnivariateFitter' is used as a name.
            Default is None.
        """
        self._data: np.ndarray = check_univariate_data(data)
        self._datatype = check_array_datatype(self._data)
        self._data = self._data.astype(self._datatype)
        self._fitted: bool = False
        self._fitted_dists: dict = {}
        self._fitted_summaries: dict = {}
        super().__init__(name)

    def __str__(self):
        return f"UnivariateFitter(fitted={self._fitted})"

    def __repr__(self):
        return self.__str__()

    def _fit_single_dist(self, name: str, fit_data: np.ndarray, data_type
                         ) -> Union[tuple, None]:
        """Fits a single distribution to the dataset."""
        try:
            # retrieving a distribution
            dist = eval(name)

            if (dist.X_DATA_TYPE == int) and (data_type == float):
                logging.warning(f'Cannot fit discrete {name} distribution to '
                                f'continuous data.')
                return None
            warnings.filterwarnings("ignore")
            fitted = dist.fit(fit_data)
            summary: pd.DataFrame = fitted.summary
            summary.columns = [fitted.name]
            return fitted, summary
        except Exception as e:
            logging.warning(f"Unable to fit {name} distribution.\n{e}")
            return None

    def _get_distributions(self, distributions: Iterable, multimodal: bool,
                           numerical: bool) -> set:
        """Gets the distribution satisfying the user's constraints."""
        distributions: set = set(distributions)
        res: set = set()
        for dist in distributions:
            if not isinstance(dist, str):
                raise TypeError("all distributions must be strings.")
            dist: str = dist.lower()

            if dist in distributions_map.keys():
                res = res.union(distributions_map[dist])
            else:
                if dist in distributions_map['all']:
                    res = res.union([dist])
                else:
                    logging.warning(f"{dist} is not an implemented "
                                    f"distribution. Check naming is correct.")

        if not multimodal:
            # removing all multimodal distributions
            res = res.difference(distributions_map['all multimodal'])
        if not numerical:
            # removing all numerical distributions
            res = res.difference(distributions_map['all numerical'])
        return set(d.replace('-', '_') for d in res)

    def _fit_processpoolexecutor(self, distributions: set, func: Callable,
                                 timeout):
        """Fits distributions using ProcessPoolExecutor."""
        remaining: int = len(distributions)
        if remaining == 0:
            # early stopping
            return None

        # fitting
        with ProcessPoolExecutor() as executor:
            results = executor.map(func, distributions, timeout=timeout)

        # processing results

        while remaining > 0:
            remaining -= 1
            try:
                for res in results:
                    if res is not None:
                        self._fitted_dists[res[0].name] = res[0]
                        self._fitted_summaries[res[0].name] = res[1]
            except PicklingError:
                pass

    def _fit_sequentially(self, distributions: set, func: Callable):
        """Fits distributions non-concurrently."""
        for name in distributions:
            res = func(name)
            if res is not None:
                self._fitted_dists[res[0].name] = res[0]
                self._fitted_summaries[res[0].name] = res[1]

    def fit(self, distributions: Union[str, Iterable] = None, data_type=None,
            multimodal: bool = False, numerical: bool = False,
            timeout: int = 10, raise_error: bool = False,
            use_processpoolexecutor: bool = False, **kwargs):
        """Fits the specified probability distributions to the data.

        Parameters
        ----------
        distributions: Union[Iterable, str, None]
            The distributions specified by the user.
            If an iterable, it must contain the string names of the
            distributions the user wants to fit and/or the distribution
            categories to use.
            If a string, it must be a valid category of distributions or name
            of a distribution to fit.
            Distribution categories are: 'all', 'all continuous',
            'all discrete', 'all common', 'all multimodal', 'all parametric',
            'all numerical', 'all continuous parametric',
            'all discrete parametric', 'all continuous numerical',
            'all discrete numerical', 'common continuous', 'common discrete',
            'continuous multimodal', 'discrete multimodal'.
            If None, either 'common discrete' or 'common continuous' will be
            used depending on the given data. Note, if a multimodal or
            numerical category/distribution is specified,
            the multimodal / numerical arguments must also be set to True.
            Default is None.
        data_type: Union[str, None]
            The data-type of the distributions to fit if the distributions
            argument is not given. float for continuous distributions and int
            for discrete distributions. Default is None.
        multimodal: bool
            Whether to include multimodal distributions when fitting if the
            distributions argument is not given.
            Default is False. Must be True if a multimodal distribution or
            category is specified in the distributions argument.
        numerical: bool
            Whether to include numerical distributions when fitting if the
            distributions argument is not given.
            Default is False. Must be True if a numerical distribution or
            category is specified in the distributions argument.
        timeout: int
            The maximum amount of time (seconds) to fit each distribution.
            Default is 10.
        raise_error: bool
            Whether to raise an error if no distributions are fitted.
            Default is False.
        use_processpoolexecutor: bool
            Whether to use ProcessPoolExecutor to fit distributions
            concurrently.
            Note that, if code is not run inside
            `if __name__ == '__main__': ... `
            in the main module, you may receive a runtime error.
            Default is False.

        Returns
        -------
        self
            A fitted UnivariateFitter object.
        """

        # argument checks
        if data_type is None:
            data_type = self._datatype
        elif not ((data_type == int) or (data_type == float)):
            raise ValueError("data_type must be int or float")

        if distributions is None:
            if data_type is int:
                distributions = ['common discrete']
            elif data_type is float:
                distributions = ['common continuous']
            else:
                distributions = ['all common']
        elif isinstance(distributions, str):
            distributions = [distributions]
        elif isinstance(distributions, Iterable):
            if len(distributions) < 1:
                raise ValueError('At least one distribution must be '
                                 'specified.')
        else:
            raise TypeError("Distributions must be a string or iterable.")

        for bool_arg in (multimodal, numerical, raise_error,
                         use_processpoolexecutor):
            if not isinstance(bool_arg, bool):
                raise TypeError("multimodal, numerical, raise_error, "
                                "use_processpoolexecutor arguments must be "
                                "bool.")

        if isinstance(timeout, int):
            if timeout <= 0:
                raise ValueError("timeout must be a positive integer.")
        else:
            raise TypeError("timeout must be a positive integer.")

        # putting data in the correct data-type
        fit_data: np.ndarray = self._data.astype(data_type)

        # getting list of distributions
        distributions: set = self._get_distributions(distributions,
                                                     multimodal, numerical)

        # fitting distributions
        func: Callable = partial(self._fit_single_dist, fit_data=fit_data,
                                 data_type=data_type)
        if use_processpoolexecutor:
            ppe_distributions: set = distributions.intersection(
                distributions_map['all parametric'])
            distributions: set = distributions.difference(ppe_distributions)
            self._fit_processpoolexecutor(ppe_distributions, func, timeout)
        self._fit_sequentially(distributions, func)

        if len(self._fitted_dists) == 0:
            msg: str = "Unable to fit any distribution to the data."
            if raise_error:
                raise FitError(msg)
            else:
                logging.warning(msg)
        self._fitted = True
        return self

    def _filter(self, pvalue: float) -> tuple:
        """method to filter out all distributions which do not statistically
        fit the data for a given p-value."""
        if not self._fitted:
            raise FitError("UnivariateFitter has not been fitted to data. "
                           "Call .fit method.")

        filtered_dists: dict = {}
        filtered_summaries: dict = {}
        for name in self._fitted_dists:
            dist = self._fitted_dists[name]
            gof: pd.DataFrame = dist.gof()

            if dist.continuous_or_discrete == 'continuous':
                # Cramér-von Mises and Kolmogorov-Smirnov gof tests
                # for continuous distributions
                cvm_pvalue: float = gof.loc[
                    'Cramér-von Mises p-value'].values[0]
                ks_pvalue: float = gof.loc[
                    'Kolmogorov-Smirnov p-value'].values[0]

                # keeping only the distributions which are significant
                # for BOTH tests
                if (cvm_pvalue >= pvalue) and (ks_pvalue >= pvalue):
                    filtered_dists[name] = dist
                    filtered_summaries[name] = self._fitted_summaries[name]

            elif dist.continuous_or_discrete == 'discrete':
                # Chi-squared gof tests for discrete distributions
                chisq_pvalue: float = gof.loc['chi-square p-value'].values[0]

                if chisq_pvalue >= pvalue:
                    filtered_dists[name] = dist
                    filtered_summaries[name] = self._fitted_summaries[name]
        return filtered_dists, filtered_summaries

    def get_summary(self, sortby: str = None, significant: bool = False,
                    pvalue: float = 0.05) -> pd.DataFrame:
        """Returns a summary of the fitted distributions.

        Parameters
        ----------
        sortby: str
            The metric/column to sort the summary by. None to not sort.
            Default is None.
        significant: bool
            True to remove distributions which fail goodness of fit
            significance tests.
            Default is False.
        pvalue: float
            The p-value to use when rejecting the null hypothesis in goodness
            of fit tests.
            Default is 0.05.

        Returns
        -------
        summary: pd.DataFrame
            The summary of fitted distributions.
        """
        if not self._fitted:
            raise FitError("UnivariateFitter has not been fitted to data. "
                           "Call .fit method.")

        # argument checks
        if not isinstance(significant, bool):
            raise TypeError("significant must be a boolean.")

        if not (isinstance(pvalue, float) or isinstance(pvalue, int)):
            raise TypeError("pvalue must be a float.")
        elif (pvalue > 1) or (pvalue < 0):
            raise ValueError("pvalue must be a valid probability.")

        if significant:
            _, summaries = self._filter(pvalue)
        else:
            summaries = self._fitted_summaries

        if len(summaries) == 0:
            # No distributions fitted successfully
            return pd.DataFrame()

        summary: pd.DataFrame = pd.concat(summaries.values(), axis=1
                                          ).transpose()
        max_num_params: int = summary['#Params'].max()
        if max_num_params != 0:
            # putting columns in desired order
            cols: list = summary.columns.to_list()
            param0_index: int = cols.index('param0')
            param_cols: list = [f'param{i}' for i in range(max_num_params)]
            non_param_cols: list = [col for col in cols
                                    if col not in param_cols]
            new_order: list = [
                *cols[:param0_index],
                *param_cols,
                *non_param_cols[param0_index:]
            ]
            summary = summary[new_order]

        if (sortby is None) or (sortby not in summary.columns):
            return summary
        return summary.sort_values(by=sortby)

    def get_best(self, significant: bool = True, pvalue: float = 0.05,
                 raise_error: bool = False, **kwargs):
        """Returns the fitted probability distribution which minimises the sum
        of squared error between the empirical and fitted pdfs.

        Parameters
        ----------
        significant: bool
            True to select only from distributions which satisfy goodness of
            fit significance tests.
            Default is True.
        pvalue: float
            The p-value to use when rejecting the null hypothesis in
            goodness of fit tests.
            Default is 0.05.
        raise_error: bool
            True to raise an error if no distribution can be returned.
            If False and no distribution can be returned, an empirical
            distribution is fitted and returned.
            Default is False.

        Returns
        -------
        best:
            best fitted distribution.
        """
        # argument checks
        if not isinstance(raise_error, bool):
            raise TypeError("raise_error must be a boolean.")

        summary: pd.DataFrame = self.get_summary('Sum of Squared Error',
                                                 significant, pvalue)
        if len(summary) == 0:
            # no good distributional fits
            if raise_error:
                raise SignificanceError("No statistically significant "
                                        "distributions fitted.")
            logging.warning("No statistically significant distributions "
                            "fitted. Empirical distribution returned.")

            # fitting numerical distributions
            if self._datatype == int:
                return discrete_empirical.fit(self._data)
            return empirical.fit(self._data)

        best = summary.index[0]
        return self._fitted_dists[best]

    def plot(self, which: Union[str, Iterable] = 'all', pvalue: float = 0.05,
             xrange: np.ndarray = None, include_empirical: bool = True,
             empirical_color: str = 'black', qqplot_yx_color: str = 'black',
             alpha: float = 0.5, empirical_alpha: float = 1.0,
             qqplot_yx_alpha: float = 1.0, pdf_plot_limit: float = 1.0,
             figsize: tuple = (16, 8), grid: bool = True,
             num_to_plot: int = 100, show: bool = True) -> None:
        """Plots the fitted distributions. Produces subplots of the pdf and
        cdf.

        Parameters
        ----------
        which: Union[str, Iterable]
            'all', 'best', 'significant', 'best significant' to plot all,
            the best and significant fitted distributions. which can also be
            the string name of a fitted distribution.
            If an iterable, the distribution names which are specified in the
            iterable and have been fitted are plotted.
            Default is 'all'.
        pvalue: float
            The p-value to use when rejecting the null hypothesis in goodness
            of fit tests. Used if which is 'significant' or 'best significant'.
            Default is 0.05.
        xrange: np.ndarray
            A user supplied range to plot the distribution
            (and empirical distribution) over.
            If not provided, this will be generated.
        include_empirical: bool
            Whether to include empirical distribution in your plots.
            Default is True.
        empirical_color: str
            The color in which to plot the empirical distribution.
            Any acceptable value for the matplotlib.pyplot 'color' argument
            can be given.
            Default is 'black'.
        qqplot_yx_color: str
            The color in which to plot the y=x line in the QQ-plot. Any
            acceptable value for the matplotlib.pyplot 'color' argument can be
            given.
            Default is 'black'.
        pdf_plot_limit: float
            The upper limit for pdf values to use in plots. This may be useful
            as fitted distributions may sometimes have unreasonably large pdf
            values which skew the proportions of the plots, making analysis
            difficult.
            Default is 1.0.
        alpha: float
            The alpha/transparency value to use when plotting the distribution.
            Default is 0.5
        empirical_alpha: float
            The alpha/transparency value to use when plotting the empirical
            distribution.
            Default is 1.0.
        qqplot_yx_alpha: float
            The alpha/transparency value to use when plotting the y=x line in
            the QQ-plot.
        figsize: tuple
            The size/dimensions of the figure.
            Default is (16, 8).
        grid: bool
            Whether to include a grid in the plots.
            Default is True.
        num_to_plot: int
            The number of points to plot.
            Default is 100.
        show: bool
            Whether to show the plots.
            Default is True.
        """
        if not self._fitted:
            raise FitError("UnivariateFitter has not been fitted to data. "
                           "Call .fit method.")

        # argument checks
        if isinstance(which, str):
            which = which.lower()
            if which == 'all':
                dists = self.get_summary().index
            elif which == 'significant':
                summary: pd.DataFrame = self.get_summary(significant=True,
                                                         pvalue=pvalue)
                if len(summary) == 0:
                    raise SignificanceError("No statistically significant "
                                            "distributions fitted.")
                dists = summary.index
            elif which == 'best':
                dists = [self.get_best().name]
            elif which == 'best significant':
                dists = [self.get_best(significant=True, pvalue=pvalue).name]
            elif which in self._fitted_dists.keys():
                dists = [which]
            else:
                raise ValueError(f"if which is a string, it must be 'all', "
                                 f"'significant', 'best', 'best significant' "
                                 f"or the name of a fitted distribution, "
                                 f"not '{which}'")
        elif isinstance(which, Iterable):
            dists = [d for d in which if d in self._fitted_dists.keys()]
        else:
            raise TypeError("which must be a string or iterable")

        if xrange is None:
            if not (isinstance(num_to_plot, int) and num_to_plot >= 1):
                raise TypeError("invalid argument type in plot. "
                                "check num_to_plot is a natural number.")
            xrange: np.ndarray = np.linspace(
                self._data.min(), self._data.max(), num_to_plot)
        elif isinstance(xrange, np.ndarray):
            if xrange.size < 1:
                raise ValueError("xrange cannot be empty.")
        else:
            raise TypeError("xrange must be None or a numpy array.")

        for bool_arg in (include_empirical, grid, show):
            if not isinstance(bool_arg, bool):
                raise TypeError("invalid argument type in plot. check "
                                "include_empirical, grid, show are all "
                                "boolean.")

        for str_arg in (empirical_color, qqplot_yx_color):
            if not isinstance(str_arg, str):
                raise TypeError("invalid argument type in plot. check "
                                "empirical_color, qqplot_yx_color are all "
                                "strings.")

        for float_arg in (alpha, empirical_alpha,
                          qqplot_yx_alpha, pdf_plot_limit):
            if not isinstance(float_arg, float):
                raise TypeError("invalid argument type in plot. check alpha, "
                                "empirical_alpha, qqplot_yx_alpha, "
                                "pdf_plot_limit are all floats.")

        if not (isinstance(figsize, tuple) and len(figsize) == 2):
            raise TypeError("invalid argument type in plot. check figsize "
                            "is a tuple of length 2.")

        # creating qrange
        qrange: np.ndarray = np.linspace(0, 1, num_to_plot, dtype=float)

        # max pdf value to plot
        max_pdf_value: float = 0.0

        # creating subplots
        xlabels: tuple = ("x", "x", "P(X<=q)", "Theoretical Quantiles")
        ylabels: tuple = ("PDF", "P(X<=x)", "q", "Empirical Quantiles")
        titles: tuple = ('PDF', 'CDF', 'Inverse CDF', "QQ-Plot")
        fig, ax = plt.subplots(1, 4, figsize=figsize)

        # fitting empirical dist for plotting
        if self._datatype == float:
            empirical_dist = empirical.fit(self._data.astype(float))
        elif self._datatype == int:
            empirical_dist = discrete_empirical.fit(self._data.astype(int))

        # plotting empirical distribution
        empirical_ppf_values: np.ndarray = empirical_dist.ppf(qrange)
        if include_empirical:
            empirical_label: str = 'Empirical'
            empirical_pdf_values: np.ndarray = empirical_dist.pdf(xrange)
            ax[0].plot(xrange, empirical_pdf_values, color=empirical_color,
                       alpha=empirical_alpha, label=empirical_label)
            ax[1].plot(xrange, empirical_dist.cdf(xrange),
                       color=empirical_color, alpha=empirical_alpha,
                       label=empirical_label)
            ax[2].plot(qrange, empirical_ppf_values, color=empirical_color,
                       alpha=empirical_alpha, label=empirical_label)

            max_pdf_value = empirical_pdf_values.max()

        # plotting distributions
        ax[3].plot(xrange, xrange, color=qqplot_yx_color,
                   alpha=qqplot_yx_alpha, label='y=x')
        for name in dists:
            dist = self._fitted_dists[name]

            pdf_values: np.ndarray = dist.pdf(xrange)
            cdf_values: np.ndarray = dist.cdf(xrange)
            ppf_values: np.ndarray = dist.ppf(qrange)

            ax[0].plot(xrange, pdf_values, alpha=alpha, label=dist.name)
            ax[1].plot(xrange, cdf_values, alpha=alpha, label=dist.name)
            ax[2].plot(qrange, ppf_values, alpha=alpha, label=dist.name)
            ax[3].scatter(ppf_values, empirical_ppf_values, label='Quartiles',
                          alpha=alpha)

            max_pdf_value = max(max_pdf_value, pdf_values.max())
        ax[0].set_ylim(0, min(pdf_plot_limit, max_pdf_value))

        # labelling axes
        for i in range(4):
            ax[i].set_xlabel(xlabels[i])
            ax[i].set_ylabel(ylabels[i])
            ax[i].set_title(titles[i])
            ax[i].grid(grid)

        # choosing legend location
        num_dists: int = len(self._fitted_dists)
        if num_dists <= 10:
            ncols: int = num_dists
        else:
            ncols: int = 10
        nrows = math.ceil(num_dists / 4)
        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=ncols,
                   bbox_to_anchor=(0.5, 0), fancybox=True, shadow=True)
        plt.tight_layout()
        bottom = max(nrows * 0.2 / figsize[1], 0.12)
        fig.subplots_adjust(bottom=bottom)

        if show:
            plt.show()

    @property
    def fitted_distributions(self) -> dict:
        """All distributions fitted to the dataset."""
        if self._fitted:
            return self._fitted_dists
        raise FitError("UnivariateFitter has not been fitted to data. "
                       "Call .fit method.")

# Contains classes for holding fitted univariate distributions
from typing import Callable, Union
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklarpy.utils._serialize import Savable

__all__ = ['FittedDiscreteUnivariate', 'FittedContinuousUnivariate']


class FittedUnivariateBase(Savable):
    """Base class for holding a fitted probability distribution."""
    _OBJ_NAME = "FittedUnivariateBase"

    def __init__(self, obj, fit_info: dict):
        """
        Class used to hold a fitted probability distribution.

        Parameters
        ----------
        obj:
            Either a PreFitContinuousUnivariate or PreFitDiscreteUnivariate
            object.
        fit_info: dict
            A dictionary containing information about the fit.
        """
        self.__obj = obj
        self.__fit_info: dict = fit_info
        Savable.__init__(self, self.__obj.name)
        self._ppf_approx: Callable = None
        self._ppf_approx_num_points: int = 100

    def __str__(self) -> str:
        """The name of the distribution + parameters"""
        return self.name_with_params

    def __repr__(self) -> str:
        """The name of the distribution + parameters"""
        return self.__str__()

    def pdf(self, x: Union[float, int, np.ndarray]) -> np.ndarray:
        """The probability density/mass function.

        Parameters
        ----------
        x: Union[float, int, np.ndarray]
            The value/values to calculate the pdf/pmf values of.
            In the case of a discrete distribution, these values are P(X=x).

        Returns
        --------
        pdf_values: np.ndarray
            An array of pdf values
        """
        return self.__obj.pdf(x, self.params)

    def cdf(self, x: Union[float, int, np.ndarray]) -> np.ndarray:
        """The cumulative distribution function.

        Parameters
        -----------
        x: Union[float, int, np.ndarray]
            The value/values to calculate the cdf values, P(X<=x), of.

        Returns
        -------
        cdf_values: np.ndarray
            An array of cdf values
        """
        return self.__obj.cdf(x, self.params)

    def ppf(self, q: Union[float, int, np.ndarray], **kwargs) -> np.ndarray:
        """The cumulative inverse function

        Parameters
        -----------
        q: Union[float, int, np.ndarray]
            The quartile values to calculate cdf^-1(q) of.

        Returns
        --------
        ppf_values: np.ndarray
            An array of quantile values.
        """
        return self.__obj.ppf(q, self.params)

    def ppf_approx(self, q: Union[float, int, np.ndarray],
                   num_points: int = 100, eps: float = 0.01, **kwargs
                   ) -> np.ndarray:
        """The approximate cumulative inverse function.
        We evaluate the ppf function on a (eps, 1-eps) linspace of quartiles.
        We then fit a linear interpolation between these values. Then, using
        this linear interpolation function, for each point in q, we calculate
        the approximate ppf. Note if a given qi lies outside (eps, 1-eps),
        we use the (non-approximate) ppf function, allowing us to accurately
        capture tail behavior still. Also, if the number of points in q lying
        inside (eps, 1-eps) if less than or equal to the num_points argument,
        we use ppf, as this will be faster. The first time this method is
        called, the linear interpolation is calculated and saved. If the user
        then calls this method again, the same linear interpolation is reused,
        allowing for faster computation. If the user changes the num points
        argument, the linear interpolation will be recalculated and saved
        again.

        Parameters
        ----------
        q: Union[float, int, np.ndarray]
            The quartile values to calculate cdf^-1(q) of.
        num_points: int
            The number of points / quartiles in a (eps, 1-eps) linspace to
            evaluate the (non-approx) ppf function at.
            Default value is 100.
        eps: float
            The epsilon value to use.

        Returns
        -------
        rvs_values: np.ndarray
            A random sample of dimension 'size'.
        """

        return self.__obj.ppf_approx(q, self.params, num_points, eps, **kwargs)

    def cdf_approx(self, x: Union[float, int, np.ndarray],
                   num_points: int = 100, **kwargs) -> np.ndarray:
        """The approximate cumulative density function.

        We evaluate the cdf function on over a linspace of values in
        (xmin, xmax) of the given data. We then fit a linear interpolation
        between these values. Then, using this linear interpolation function,
        we calculate the approximate cdf values.
        This is useful when there is no analytical cdf function, as evaluating
        many numerical integrals can be slow.

        Parameters
        ---------
        x: Union[float, int, np.ndarray]
            The value/values to calculate the cdf values, P(X<=x) of.
        num_points: int
            The number of points in the (xmin, xmax) linspace to evalute the
            (non-approx) cdf function at.
            Default is 100.

        Returns
        -------
        cdf_approx_values: np.ndarray
            An array of cdf values.
        """
        return self.__obj.cdf_approx(x, self.params, num_points, **kwargs)

    def rvs(self, size: tuple, ppf_approx: bool = False, **kwargs
            ) -> np.ndarray:
        """Random sampler.

        Parameters
        ----------
        size: tuple
            The dimensions/shape of the random variable array output.
        ppf_approx: bool
            Whether to use ppf_approx and the inverse transformation method
            for random sampling. This can be a lot faster for certain
            distributions (such as the GIG / Generalised Hyperbolic) as
            sampling using the inverse transform method and the (non-approx)
            ppf, requires numerically integrating for each rv generated,
            which can be slow when we are sampling a large number of variates.
        kwargs:
            Keyword arguments to pass to ppf_approx (if used).

        Returns
        -------
        rvs_values: np.ndarray
            A random sample of dimension 'size'.
        """
        return self.__obj.rvs(size, params=self.params, ppf_approx=ppf_approx,
                              ppf_approx_func=self.ppf_approx)

    def logpdf(self, x: Union[float, int, np.ndarray]) -> np.ndarray:
        """The logarithm of the probability density/mass function.

        Parameters
        ----------
        x: Union[float, int, np.ndarray]
            The value/values to calculate the logpdf values of.

        Returns
        -------
        logpdf_values: np.ndarray
            An array of logpdf values
        """
        return self.__obj.logpdf(x, self.params)

    def likelihood(self, data: np.ndarray = None) -> float:
        """The likelihood function.

        Parameters
        ----------
        data: np.ndarray, optional
            The data to calculate the likelihood of.
            If not provided, the likelihood value of the fitted data will be
            returned.

        Returns
        -------
        likelihood_value: float
            The value of the likelihood function.
        """
        if data is None:
            return self.__fit_info['likelihood']
        return self.__obj.likelihood(data, self.params)

    def loglikelihood(self, data: np.ndarray = None) -> float:
        """The logarithm of the likelihood function.

        Parameters
        ----------
        data: np.ndarray, optional
            The data to calculate the log-likelihood of.
            If not provided, the loglikelihood value of the fitted data will
            be returned.

        Returns
        -------
        loglikelihood_value: float
            The value of the log-likelihood function.
        """
        if data is None:
            return self.__fit_info['loglikelihood']
        return self.__obj.loglikelihood(data, self.params)

    def aic(self, data: np.ndarray = None) -> float:
        """Calculates the Akaike Information Criterion (AIC)

        Parameters
        ----------
        data: np.ndarray
            The data to calculate the AIC of.
            If not provided, the AIC value of the fitted data will be returned.

        Returns
        -------
        aic_value: float
            The value of the Akaike Information Criterion (AIC).
        """
        if data is None:
            return self.__fit_info['aic']
        return self.__obj.aic(data, self.params)

    def bic(self, data: np.ndarray = None) -> float:
        """Calculates the Bayesian Information Criterion (BIC)

        Parameters
        ----------
        data: np.ndarray
            The data to calculate the BIC of.
            If not provided, the BIC value of the fitted data will be returned.

        Returns
        -------
        bic_value: float
            The value of the Bayesian Information Criterion (BIC).
        """
        if data is None:
            return self.__fit_info['bic']
        return self.__obj.bic(data, self.params)

    def sse(self, data: np.ndarray = None) -> float:
        """The sum of squared error between the fitted distribution and
        provided data.

        Parameters
        ----------
        data: np.ndarray
            The data to calculate the SSE of.
            If not provided, the SSE value of the fitted data will be returned.

        Returns
        -------
        sse_value: float
            The value of the sum of squared error.
        """
        if data is None:
            return self.__fit_info['sse']
        return self.__obj.sse(data, self.params)

    def gof(self, data: np.ndarray = None) -> pd.DataFrame:
        """Calculates goodness of fit tests for the specified distribution
        against data.

        Parameters
        -----------
        data: np.ndarray
            The data to compute goodness of fit tests with respect to.
            If not provided, the goodness of fit test results of the fitted
            data will be returned.

        Returns
        -------
        gof: pd.DataFrame
            A dataframe containing the test statistics and p-values of the
            goodness of fit tests.
        """
        if data is None:
            return self.__fit_info['gof']
        return self.__obj.gof(data, self.params)

    def plot(self, xrange: np.ndarray = None, include_empirical: bool = True,
             color: str = 'royalblue', empirical_color: str = 'black',
             qqplot_yx_color: str = 'black', alpha: float = 1.0,
             empirical_alpha: float = 1.0,  qqplot_yx_alpha: float = 1.0,
             figsize: tuple = (16, 8), grid: bool = True,
             num_to_plot: int = 100, show: bool = True) -> None:
        """Plots the fitted distribution. Produces subplots of the pdf, cdf,
        inverse cdf/ppf and QQ-plot.

        Parameters
        -----------
        xrange: np.ndarray
            A user supplied range to plot the distribution
            (and empirical distribution) over.
            If not provided, this will be generated.
        include_empirical: bool
            Whether to include empirical distribution in your plots. When the
            distribution has been fitted using parameters, this is False and
            the empirical distribution is not included in plots.
            Default is True.
        color: str
            The color in which to plot the fitted distribution. Any acceptable
            value for the matplotlib.pyplot 'color' argument can be given.
            Default is 'royalblue'.
        empirical_color: str
            The color in which to plot the empirical distribution. Any
            acceptable value for the matplotlib.pyplot 'color' argument can be
            given.
            Default is 'black'.
        qqplot_yx_color: str
            The color in which to plot the y=x line in the QQ-plot. Any
            acceptable value for the matplotlib.pyplot 'color' argument can be
            given.
            Default is 'black'.
        alpha: float
            The alpha/transparency value to use when plotting the distribution.
            Default is 1.0.
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
        # checking arguments
        for bool_arg in (include_empirical, grid, show):
            if not isinstance(bool_arg, bool):
                raise TypeError("invalid argument type in plot. check "
                                "include_empirical, grid, show are all "
                                "boolean.")

        for str_arg in (color, empirical_color, qqplot_yx_color):
            if not isinstance(str_arg, str):
                raise TypeError("invalid argument type in plot. check color, "
                                "empirical_color, qqplot_yx_color are all "
                                "strings.")

        for float_arg in (alpha, empirical_alpha, qqplot_yx_alpha):
            if not isinstance(float_arg, float):
                raise TypeError("invalid argument type in plot. check alpha, "
                                "empirical_alpha, qqplot_yx_alpha are all "
                                "floats.")

        if not (isinstance(figsize, tuple) and len(figsize) == 2):
            raise TypeError("invalid argument type in plot. check figsize is "
                            "a tuple of length 2.")

        if not self.fitted_to_data:
            include_empirical = False
            if include_empirical:
                logging.warning("include_empirical is true, but distribution "
                                "was not fitted on any data. Hence we have "
                                "no empirical data to display.")

        # getting xrange and qrange
        eps: float = 0.05
        prob_bounds: tuple = (eps, 1 - eps)
        if xrange is None:
            if not (isinstance(num_to_plot, int) and num_to_plot >= 1):
                raise TypeError("invalid argument type in plot. check "
                                "num_to_plot is a natural number.")
            if len(self.fitted_domain) == 2:
                # distribution was fitted using data.
                xmin, xmax = self.fitted_domain
            else:
                # distribution was fitted only using params.
                xmin = max(self.__obj.X_DATA_TYPE(self.ppf(prob_bounds[0])),
                           self.support[0])
                xmax = min(self.__obj.X_DATA_TYPE(self.ppf(prob_bounds[1])),
                           self.support[1])
            xrange = np.linspace(xmin, xmax, num_to_plot,
                                 dtype=self.__obj.X_DATA_TYPE)
        # elif isinstance(xrange, np.ndarray):
        #     if xrange.size < 1:
        #         raise ValueError("xrange cannot be empty.")
        else:
            if not (isinstance(xrange, np.ndarray) and (xrange.size >= 1)):
                raise TypeError("invalid argument type in plot. check xrange "
                                "is a np.ndarray and is non-empty.")

        # creating qrange
        qrange: np.ndarray = np.linspace(*prob_bounds, num_to_plot)

        # creating subplots
        num_subplots = 3 + include_empirical
        fig, ax = plt.subplots(1, num_subplots, figsize=figsize)
        xlabels: tuple = ("x", "x", "P(X<=q)", "Theoretical Quantiles")
        ylabels: tuple = ("PDF", "P(X<=x)", "q", "Empirical Quantiles")
        titles: tuple = ('PDF', 'CDF', 'Inverse CDF', "QQ-Plot")

        # doing empirical plots
        if include_empirical:
            # getting empirical distribution for plotting
            empirical_pdf: Callable = self.__fit_info['empirical_pdf']
            empirical_cdf: Callable = self.__fit_info['empirical_cdf']
            empirical_ppf: Callable = self.__fit_info['empirical_ppf']
            empirical_ppf_values: np.ndarray = empirical_ppf(qrange)

            # plotting empirical distribution
            empirical_label: str = 'Emprirical'
            ax[0].plot(xrange, empirical_pdf(xrange), color=empirical_color,
                       alpha=empirical_alpha, label=empirical_label)
            ax[1].plot(xrange, empirical_cdf(xrange), color=empirical_color,
                       alpha=empirical_alpha, label=empirical_label)
            ax[2].plot(qrange, empirical_ppf_values, color=empirical_color,
                       alpha=empirical_alpha, label=empirical_label)
            ax[3].plot(xrange, xrange, color=qqplot_yx_color,
                       alpha=qqplot_yx_alpha, label='y=x')
            ppf_values = self.ppf(qrange)
            ax[3].plot(ppf_values, empirical_ppf_values, color=color,
                       alpha=alpha, label=self.name)
            ax[3].set_xlim([ppf_values.min(), ppf_values.max()])

        # plotting distribution
        ax[0].plot(xrange, self.pdf(xrange), color=color,
                   alpha=alpha, label=self.name)
        ax[1].plot(xrange, self.cdf(xrange), color=color,
                   alpha=alpha, label=self.name)
        ax[2].plot(qrange, self.ppf(qrange), color=color,
                   alpha=alpha, label=self.name)

        # labelling axes
        for i in range(num_subplots):
            ax[i].set_xlabel(xlabels[i])
            ax[i].set_ylabel(ylabels[i])
            ax[i].set_title(titles[i])
            ax[i].grid(grid)
            ax[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                         fancybox=True, shadow=True, ncol=1)

        # plotting
        plt.tight_layout()
        if show:
            plt.show()

    @property
    def name_with_params(self) -> str:
        """The name of the distributions with parameters
        (rounded to 2 significant figures).
        In the form: dist_name(param0, param1, ...)
        """
        return f"{self.name}{tuple(round(p, 2) for p in self.params)}"

    @property
    def summary(self) -> pd.DataFrame:
        """A dataframe containing summary information of the distribution fit.
        """
        gof: pd.DataFrame = self.gof()
        index: list = [
            'Parametric/Non-Parametric', 'Discrete/Continuous', 'Distribution',
            '#Params', *[f"param{i}" for i in range(len(self.params))],
            'Support', 'Fitted Domain', *gof.index, 'Likelihood',
            'Log-Likelihood', 'AIC', 'BIC', 'Sum of Squared Error',
            '#Fitted Data Points'
        ]
        data: list = [
            self.__fit_info['parametric'], self.continuous_or_discrete,
            self.name, self.num_params, *self.params, self.support,
            self.fitted_domain, *gof[gof.columns[0]], self.likelihood(),
            self.loglikelihood(), self.aic(), self.bic(), self.sse(),
            self.fitted_num_data_points
        ]
        return pd.DataFrame(data, index=index, columns=['summary'])

    @property
    def params(self) -> tuple:
        """The parameters of the distribution, contained in a tuple."""
        return self.__fit_info['params']

    @property
    def support(self) -> tuple:
        """The support of the distribution."""
        return self.__fit_info['support']

    @property
    def fitted_domain(self) -> tuple:
        """The fitted domain of the distribution."""
        return self.__fit_info['fitted_domain']

    @property
    def fitted_to_data(self) -> bool:
        """Whether the distribution was fitted to a data-set (True) or fitted
        using user provided parameters (False)"""
        return self.__fit_info['fitted_to_data']

    @property
    def num_params(self) -> int:
        """The number of parameters specifying the distribution."""
        return self.__fit_info['num_params']

    @property
    def fitted_num_data_points(self) -> int:
        """The number of data points used to fit the distribution."""
        return self.__fit_info['num_data_points']

    @property
    def continuous_or_discrete(self) -> str:
        """returns 'continuous' or 'discrete'."""
        return self.__obj.continuous_or_discrete


class FittedContinuousUnivariate(FittedUnivariateBase):
    """Holds a fitted continuous univariate distribution."""


class FittedDiscreteUnivariate(FittedUnivariateBase):
    """Holds a fitted discrete univariate distribution."""

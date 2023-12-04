# Contains classes for fitting probability distributions
from typing import Callable, Union, Iterable
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import scipy.interpolate

from sklarpy.univariate._goodness_of_fit import continuous_gof, discrete_gof
from sklarpy.univariate._inverse_transform import inverse_transform
from sklarpy.utils._input_handlers import univariate_num_to_array, \
    check_univariate_data, check_array_datatype, check_params
from sklarpy.utils._errors import FitError
from sklarpy.univariate._fitted_dists import FittedContinuousUnivariate, \
    FittedDiscreteUnivariate
from sklarpy.univariate._distributions import discrete_empirical_fit, \
    continuous_empirical_fit

__all__ = [
    'PreFitParametricContinuousUnivariate',
    'PreFitParametricDiscreteUnivariate',
    'PreFitNumericalContinuousUnivariate',
    'PreFitNumericalDiscreteUnivariate'
]


class PreFitUnivariateBase:
    """Base class used to fit a univariate probability distribution."""
    _FIT_TO: Callable
    X_DATA_TYPE = None
    _PARAMETRIC: str

    def __init__(self, name: str, pdf: Callable, cdf: Callable, ppf: Callable,
                 support: Callable, fit: Callable, rvs: Callable = None):
        """Class used to fit a univariate probability distribution

        Parameters
        ----------
        name: str
            Name of the univariate distribution.
        pdf: Callable
            The pdf function for the distribution. Must take a flattened numpy
            array containing variable values and a tuple containing the
            distribution's parameters as arguments, returning a numpy array of
            pdf values.
        cdf: Callable
            The cdf function for the distribution. Must take a flattened numpy
            array containing variable values and a tuple containing the
            distribution's parameters as arguments, returning a numpy array of
            cdf values.
        ppf: Callable
            The ppf function for the distribution. Must take a flattened numpy
            array containing quartile values and a tuple containing the
            distribution's parameters as arguments, returning a numpy array of
            ppf/cdf inverse values.
        support: Callable
            The support function for the distribution. Must take a tuple
            containing the distribution's parameters as arguments, returning a
            tuple of the fitted distribution's support.
        fit: Callable
            The fit function of the distribution. Must take a flattened array
            containing a sample of data, returning a tuple containing the
            parameters of the distribution which best fits it to the data.
        rvs: Callable
            The rvs function of the distribution, used to generate random
            samples from the distribution. Must take a tuple containing the
            size/dimension of the desired random sample and a tuple containing
            the distribution's parameters as arguments, returning a numpy array
            containing the random sample of dimension 'size'. If no random
            sampler function is specified, this is implemented using the
            inverse transform method.
        """
        # argument checks
        if not isinstance(name, str):
            raise TypeError("name must be a string")

        for func in (pdf, cdf, ppf, support, fit):
            if not callable(func):
                raise TypeError("Invalid argument in pre-fit distribution "
                                "initialisation.")

        self._name: str = name
        self._pdf: Callable = pdf
        self._cdf: Callable = cdf
        self._ppf: Callable = ppf
        self._support: Callable = support
        self._fit: Callable = fit

        if rvs is None:
            rvs = partial(inverse_transform, ppf=self.ppf)
        elif not callable(rvs):
            raise TypeError("Invalid argument in pre-fit distribution "
                            "initialisation.")

        self._rvs: Callable = rvs
        self._gof: Callable = None

    def __str__(self) -> str:
        return f"PreFit{self._name.title()}Distribution"

    def __repr__(self) -> str:
        return self.__str__()

    def pdf(self, x: Union[float, int, np.ndarray], params: tuple
            ) -> np.ndarray:
        """The probability density/mass function.

        Parameters
        ----------
        x: Union[float, int, np.ndarray]
            The value/values to calculate the pdf/pmf values of.
            In the case of a discrete distribution, these values are P(X=x).
        params: tuple
            The parameters which define the univariate model.
            See scipy.stats for the correct order.

        See Also
        --------
        scipy.stats

        Returns
        -------
        pdf_values: np.ndarray
            An array of pdf values
        """
        x: np.ndarray = univariate_num_to_array(x)
        params: tuple = check_params(params)
        pdf_values: np.ndarray = self._pdf(x, *params)
        return np.where(~np.isnan(pdf_values), pdf_values, 0.0)

    def cdf(self, x: Union[float, int, np.ndarray], params: tuple
            ) -> np.ndarray:
        """The cumulative distribution function.

        Parameters
        ---------
        x: Union[float, int, np.ndarray]
            The value/values to calculate the cdf values, P(X<=x) of.
        params: tuple
            The parameters which define the univariate model.
            See scipy.stats for the correct order.

        See Also
        --------
        scipy.stats

        Returns
        -------
        cdf_values: np.ndarray
            An array of cdf values
        """
        x: np.ndarray = univariate_num_to_array(x)
        params: tuple = check_params(params)
        return self._cdf(x, *params)

    def ppf(self, q: Union[float, int, np.ndarray], params: tuple, **kwargs
            ) -> np.ndarray:
        """The cumulative inverse / quartile function.

        Parameters
        ----------
        q: Union[float, int, np.ndarray]
            The quartile values to calculate cdf^-1(q) of.
        params: tuple
            The parameters which define the univariate model.
            See scipy.stats for the correct order.

        See Also
        --------
        scipy.stats

        Returns
        -------
        ppf_values: np.ndarray
            An array of quantile values.
        """
        q: np.ndarray = univariate_num_to_array(q)
        params: tuple = check_params(params)
        return self._ppf(q, *params)

    def support(self, params: tuple) -> tuple:
        """The support function of the distribution.

        Parameters
        ----------
        params: tuple
            The parameters which define the univariate model.
            See scipy.stats for the correct order.

        See Also
        --------
        scipy.stats

        Returns
        --------
        support: tuple
            The support of the specified distribution.
        """
        params: tuple = check_params(params)
        return self._support(*params)

    def _fit_ppf_approx(self, params: tuple, num_points: int, eps: float = 0.01
                        ) -> Callable:
        if not isinstance(num_points, int) and num_points > 1:
            raise TypeError("num_points must be a positive integer greater "
                            "than 1.")
        if not isinstance(eps, float) and (eps >= 0.0) and (eps <= 1.0):
            raise TypeError("eps must be a float between 0.0 and 1.0")

        # fitting linear interpolator
        q_: np.ndarray = np.linspace(eps, 1 - eps, num_points, dtype=float)
        ppf_q_vals: np.ndarray = self.ppf(q_, params)
        return scipy.interpolate.interp1d(q_, ppf_q_vals, 'linear',
                                          bounds_error=False)

    def ppf_approx(self, q: Union[float, int, np.ndarray], params: tuple,
                   num_points: int = 100, eps: float = 0.01, **kwargs
                   ) -> np.ndarray:
        """The approximate cumulative inverse / quartile function.

        We evaluate the ppf function on a (eps, 1-eps) linspace of quartiles.
        We then fit a linear interpolation between these values. Then, using
        this linear interpolation function, for each point in q, we calculate
        the approximate ppf. Note if a given qi lies outside (eps, 1-eps), we
        use the (non-approximate) ppf function, allowing us to accurately
        capture tail behavior still. Also, if the number of points in q lying
        inside (eps, 1-eps) is less than or equal to the num_points argument,
        we use ppf, as this will be faster.

        Parameters
        ----------
        q: Union[float, int, np.ndarray]
            The quartile values to calculate cdf^-1(q) of.
        params: tuple
            The parameters which define the univariate model.
            See scipy.stats for the correct order.
        num_points: int
            The number of points / quartiles in a (eps, 1-eps) linspace to
            evaluate the (non-approx) ppf function at.
            Default value is 100.
        eps: float
            The epsilon value to use.

        See Also
        --------
        scipy.stats

        Returns
        -------
        ppf_approx_values: np.ndarray
            An array of quantile values.
        """
        q: np.ndarray = univariate_num_to_array(q)

        if ((q >= eps) & (q <= 1-eps)).sum() <= num_points:
            # faster to use ppf directly
            return self.ppf(q, params)

        # fitting ppf approx function
        ppf_approx: Callable = self._fit_ppf_approx(params, num_points, eps)

        return np.array([ppf_approx(qi) if ((qi >= eps) and (qi <= 1-eps))
                         else float(self.ppf(qi, params)) for qi in q
                         ], dtype=float)

    def _fit_cdf_approx(self, params: tuple, num_points: int, bounds: tuple
                        ) -> Callable:
        if not isinstance(num_points, int) and num_points > 1:
            raise TypeError("num_points must be a positive integer greater "
                            "than 1.")

        # fitting linear interpolator
        x_: np.ndarray = np.linspace(*bounds, num_points, dtype=float)
        cdf_x_vals: np.ndarray = self.cdf(x_, params)
        return scipy.interpolate.interp1d(x_, cdf_x_vals, 'linear',
                                          bounds_error=False)

    def cdf_approx(self, x: Union[float, int, np.ndarray], params: tuple,
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
        params: tuple
            The parameters which define the univariate model.
            See scipy.stats for the correct order.
        num_points: int
            The number of points in the (xmin, xmax) linspace to evalute the
            (non-approx) cdf function at.
            Default is 100.

        See Also
        --------
        scipy.stats

        Returns
        -------
        cdf_approx_values: np.ndarray
            An array of cdf values.
        """
        x: np.ndarray = univariate_num_to_array(x)
        params: tuple = check_params(params)

        if x.size <= num_points:
            # faster to use cdf directly
            return self.cdf(x, params)

        # fitting cdf approx function
        bounds: tuple = (x.min(), x.max())
        cdf_approx = self._fit_cdf_approx(params, num_points, bounds)
        return np.array([cdf_approx(xi) for xi in x], dtype=float)

    def rvs(self, size: tuple, params: tuple, ppf_approx: bool = False,
            **kwargs) -> np.ndarray:
        """Random sampler.

        Parameters
        ----------
        size: tuple
            The dimensions/shape of the random variable array output.
        params: tuple
            The parameters which define the univariate model.
            See scipy.stats for the correct order.
        ppf_approx: bool
            Whether to use ppf_approx and the inverse transformation method
            for random sampling. This can be a lot faster for certain
            distributions (such as the GIG / Generalised Hyperbolic) as
            sampling using the inverse transform method and the (non-approx)
            ppf, requires numerically integrating for each rv generated, which
            can be slow when we are sampling a large number of variates.
        kwargs:
            Keyword arguments to pass to ppf_approx (if used).

        See Also
        --------
        scipy.stats

        Returns
        -------
        rvs_values: np.ndarray
            A random sample of dimension 'size'.
        """
        if not isinstance(size, tuple):
            raise TypeError("size must be a tuple.")
        elif len(size) < 1:
            raise ValueError("size must not be empty.")
        if not isinstance(ppf_approx, bool):
            raise TypeError("ppf_approx must be boolean.")
        params: tuple = check_params(params)
        ppf_approx_func: Callable = kwargs.pop('ppf_approx_func',
                                               self.ppf_approx)
        return self._rvs(*params, size=size) if not ppf_approx \
            else inverse_transform(*params, size=size,
                                   ppf=ppf_approx_func, **kwargs)

    def logpdf(self, x: Union[float, int, np.ndarray], params: tuple
               ) -> np.ndarray:
        """The logarithm of the probability density/mass function.

        Parameters
        ----------
        x: Union[float, int, np.ndarray]
            The value/values to calculate the logpdf values of.
        params: tuple
            The parameters which define the univariate model.
            See scipy.stats for the correct order.

        See Also
        --------
        scipy.stats

        Returns
        -------
        logpdf_values: np.ndarray
            An array of logpdf values
        """
        return np.log(self.pdf(x, params))

    def likelihood(self, data: np.ndarray, params: tuple) -> float:
        """The likelihood function.

        Parameters
        -----------
        data: np.ndarray, optional
            The data to calculate the likelihood of.
        params: tuple
            The parameters which define the univariate model.
            See scipy.stats for the correct order.

        See Also
        --------
        scipy.stats

        Returns
        -------
        likelihood_value: float
            The value of the likelihood function.
        """
        data = check_univariate_data(data)
        pdf_values: np.ndarray = self.pdf(data, params)
        if np.any(np.isinf(pdf_values)):
            return np.inf
        return float(np.product(pdf_values))

    def loglikelihood(self, data: np.ndarray, params: tuple) -> float:
        """The logarithm of the likelihood function.

        Parameters
        ----------
        data: np.ndarray, optional
            The data to calculate the log-likelihood of.
        params: tuple
            The parameters which define the univariate model.
            See scipy.stats for the correct order.

        See Also
        --------
        scipy.stats

        Returns
        -------
        loglikelihood_value: float
            The value of the log-likelihood function.
        """
        data = check_univariate_data(data)
        logpdf_values: np.ndarray = self.logpdf(data, params)
        if np.any(np.isinf(logpdf_values)):
            # returning -np.inf instead of nan
            return -np.inf
        return float(np.sum(logpdf_values))

    def aic(self, data: np.ndarray, params: tuple) -> float:
        """Calculates the Akaike Information Criterion (AIC)

        Parameters
        ----------
        data: np.ndarray
            The data to calculate the AIC of.
        params: tuple
            The parameters which define the univariate model.
            See scipy.stats for the correct order.

        See Also
        --------
        scipy.stats

        Returns
        -------
        aic_value: float
            The value of the Akaike Information Criterion (AIC).
        """
        loglikelihood: float = self.loglikelihood(data, params)
        return 2 * (len(params) - loglikelihood)

    def bic(self, data: np.ndarray, params: tuple) -> float:
        """Calculates the Bayesian Information Criterion (BIC)

        Parameters
        ---------
        data: np.ndarray
            The data to calculate the BIC of.
        params: tuple
            The parameters which define the univariate model.
            See scipy.stats for the correct order.

        See Also
        --------
        scipy.stats

        Returns
        -------
        bic_value: float
            The value of the Bayesian Information Criterion (BIC).
        """
        loglikelihood: float = self.loglikelihood(data, params)
        return -2 * loglikelihood + np.log(data.size) * len(params)

    def sse(self, data: np.ndarray, params: tuple) -> float:
        """The sum of squared error between the fitted distribution and
        provided data.

        Parameters
        ----------
        data: np.ndarray
            The data to calculate the SSE of.
        params: tuple
            The parameters which define the univariate model.
            See scipy.stats for the correct order.

        See Also
        --------
        scipy.stats

        Returns
        -------
        sse_value: float
            The value of the sum of squared error.
        """
        pdf_values: np.ndarray = self.pdf(data, params)
        empirical_fit: Callable = self._get_empirical_fit()
        empirical_pdf, _, _, _, _ = empirical_fit(data)
        empirical_pdf_values: np.ndarray = empirical_pdf(data)
        return float(np.sum((pdf_values - empirical_pdf_values) ** 2))

    def gof(self, data, params: tuple) -> pd.DataFrame:
        """Calculates goodness of fit tests for the specified distribution
        against data.

        Parameters
        ----------
        data: np.ndarray
            The data to compute goodness of fit tests with respect to.
        params: tuple
            The parameters which define the univariate model.
            See scipy.stats for the correct order.

        See Also
        --------
        scipy.stats

        Returns
        --------
        gof: pd.DataFrame
            A dataframe containing the test statistics and p-values of the
            goodness of fit tests.
        """
        data = check_univariate_data(data)
        params = check_params(params)
        return self._gof(data, params)

    def plot(self, params: tuple, xrange: np.ndarray = None,
             color: str = 'royalblue', alpha: float = 1.0,
             figsize: tuple = (16, 8), grid: bool = True,
             num_to_plot: int = 100, show: bool = True) -> None:
        """Plots the fitted distribution. Produces subplots of the pdf, cdf,
        inverse cdf/ppf and QQ-plot.

        Parameters
        ----------
        params: tuple
            The parameters which define the univariate model.
            See scipy.stats for the correct order.
        xrange: np.ndarray
            A user supplied range to plot the distribution over.
            If not provided, this will be generated.
        color: str
            The color in which to plot the fitted distribution.
            Any acceptable value for the matplotlib.pyplot 'color' argument
            can be given.
            Default is 'royalblue'.
        alpha: float
            The alpha/transparency value to use when plotting the distribution.
            Default is 1.0.
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

        See Also
        --------
        scipy.stats
        """
        # checking arguments
        params = check_params(params)

        if not isinstance(color, str):
            raise TypeError('invalid argument type in plot. color must be a '
                            'string')

        if not isinstance(alpha, float):
            raise TypeError('invalid argument type in plot. alpha must be a '
                            'float')

        if not (isinstance(figsize, tuple) and len(figsize) == 2):
            raise TypeError("invalid argument type in plot. check figsize is "
                            "a tuple of length 2.")

        for bool_arg in (grid, show):
            if not isinstance(bool_arg, bool):
                raise TypeError("invalid argument type in plot. check grid, "
                                "show are all boolean.")

        # getting xrange and qrange
        eps: float = 0.05
        prob_bounds: tuple = (eps, 1 - eps)
        if xrange is None:
            if not (isinstance(num_to_plot, int) and num_to_plot >= 1):
                raise TypeError("invalid argument type in plot. check "
                                "num_to_plot is a natural number.")

            support: tuple = self.support(params)
            xmin = max(self.X_DATA_TYPE(self.ppf(prob_bounds[0], params)),
                       support[0])
            xmax = min(self.X_DATA_TYPE(self.ppf(prob_bounds[1], params)),
                       support[1])
            xrange = np.linspace(xmin, xmax, num_to_plot,
                                 dtype=self.X_DATA_TYPE)
        elif isinstance(xrange, np.ndarray):
            if xrange.size < 1:
                raise ValueError("xrange cannot be empty.")
        else:
            if not (isinstance(xrange, np.ndarray) and (xrange.size >= 1)):
                raise TypeError("invalid argument type in plot. check xrange "
                                "is a np.ndarray and is non-empty.")

        # creating qrange
        qrange: np.ndarray = np.linspace(*prob_bounds, num_to_plot)

        # generating required values
        xlabels: tuple = ("x", "x", "P(X<=q)")
        ylabels: tuple = ("PDF", "P(X<=x)", "q")
        titles: tuple = ('PDF', 'CDF', 'Inverse CDF')
        name_with_params: str = f"{self.name}" \
                                f"{tuple(round(p, 2) for p in params)}"
        distribution_values: tuple = (
            (xrange, self.pdf(xrange, params)),
            (xrange, self.cdf(xrange, params)),
            (qrange, self.ppf(qrange, params)))

        # creating plots
        fig, ax = plt.subplots(1, 3, figsize=figsize)
        for i in range(3):
            # plotting distribution
            ax[i].plot(distribution_values[i][0], distribution_values[i][1],
                       label=name_with_params, color=color, alpha=alpha)

            # labelling axis
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

    def _get_empirical_fit(self):
        """returns the continuous_empirical_fit or discrete_empirical_fit
        function"""
        return eval(f"{self.continuous_or_discrete}_empirical_fit")

    def _fit_given_params(self, params: tuple
                          ) -> Union[FittedContinuousUnivariate,
                                     FittedDiscreteUnivariate]:
        """Fits the distribution using user provided parameters and not data.

        Parameters
        ----------
        params: tuple
            The parameters which define the univariate model.
            See scipy.stats for the correct order.

        See Also
        --------
        scipy.stats

        Returns
        -------
        fdist: Union[FittedContinuousUnivariate, FittedDiscreteUnivariate]
            A fitted distribution.
        """
        params = check_params(params)
        support: tuple = self.support(params)

        # returning fitted distribution
        fit_info: dict = {
            "fitted_to_data": False,
            "params": params,
            "support": support,
            "fitted_domain": (),
            "gof": pd.DataFrame([], columns=[self.name]),
            "likelihood": np.nan,
            "loglikelihood": np.nan,
            "num_data_points": 0,
            "num_params": len(params),
            "aic": np.nan,
            "bic": np.nan,
            "sse": np.nan,
            "parametric": self._PARAMETRIC,
        }

        obj = copy.deepcopy(self)
        return self._FIT_TO(obj, fit_info)

    def _calc_fit_stats(self, data: np.ndarray, params: tuple,
                        empirical_pdf_values: np.ndarray) -> tuple:
        """Calculates statistics related to the distributional fit.

        Parameters
        -----------
        data : np.ndarray
            The data to use to fit the distribution..
        params: tuple
            The parameters which define the univariate model.
            See scipy.stats for the correct order.
        empirical_pdf_values: np.ndarray
            The empirical PDF values of the data.

        See Also
        --------
        scipy.stats

        Returns
        -------
        fit_stats: tuple
            The fit statistics.
        """
        gof: pd.DataFrame = self._gof(data, params)
        pdf_values: np.ndarray = self.pdf(data, params)
        likelihood: float = float(np.product(pdf_values))
        loglikelihood: float = float(np.sum(np.log(pdf_values)))
        num_data_points: int = len(data)
        num_params: int = len(params)
        aic: float = 2 * num_params - 2 * loglikelihood
        bic: float = -2 * loglikelihood + np.log(num_data_points) * num_params
        sse: float = float(np.sum((pdf_values - empirical_pdf_values) ** 2))
        return (gof, likelihood, loglikelihood, num_data_points,
                num_params, aic, bic, sse)

    def _fit_given_data(
            self, data: Union[pd.DataFrame, pd.Series, np.ndarray, Iterable],
            params: tuple = None)\
            -> Union[FittedContinuousUnivariate, FittedDiscreteUnivariate]:
        """Fits the distribution using user provided data.

        Parameters
        ----------
        data : Union[pd.DataFrame, pd.Series, np.ndarray, Iterable]
            The data to fit to the distribution too.
            Can be a pd.DataFrame, pd.Series, np.ndarray or any other iterable
            containing data.
        params: tuple
            The parameters which define the univariate model.
            See scipy.stats for the correct order.

        See Also
        --------
        scipy.stats

        Returns
        --------
        fdist: Union[FittedContinuousUnivariate, FittedDiscreteUnivariate]
            A fitted distribution.
        """
        # checking arguments
        data: np.ndarray = check_univariate_data(data)

        if (check_array_datatype(data) == float) and (self.X_DATA_TYPE == int):
            raise FitError('Cannot fit discrete distribution to continuous '
                           'data.')

        if params is None:
            params: tuple = self._fit(data)
        else:
            params = check_params(params)

        # calculating support and fitted domain
        support: tuple = self.support(params)
        fitted_domain: tuple = (data.min(), data.max())

        # fitting empirical distribution
        empirical_fit: Callable = self._get_empirical_fit()
        empirical_pdf, empirical_cdf, empirical_ppf, _, _ = empirical_fit(data)
        empirical_pdf_values: np.ndarray = empirical_pdf(data)

        # fit statistics
        (gof, likelihood, loglikelihood, num_data_points, num_params, aic, bic,
         sse) = self._calc_fit_stats(data, params, empirical_pdf_values)

        # returning fitted distribution
        fit_info: dict = {
            "fitted_to_data": True,
            "params": params,
            "support": support,
            "fitted_domain": fitted_domain,
            "empirical_pdf": empirical_pdf,
            "empirical_cdf": empirical_cdf,
            "empirical_ppf": empirical_ppf,
            "gof": gof,
            "likelihood": likelihood,
            "loglikelihood": loglikelihood,
            "num_data_points": num_data_points,
            "num_params": num_params,
            "aic": aic,
            "bic": bic,
            "sse": sse,
            "parametric": self._PARAMETRIC,
        }

        obj = copy.deepcopy(self)
        return self._FIT_TO(obj, fit_info)

    def fit(self,
            data: Union[pd.DataFrame, pd.Series, np.ndarray, Iterable] = None,
            params: tuple = None) -> Union[FittedContinuousUnivariate,
                                           FittedDiscreteUnivariate]:
        """Used to fit the distribution to the data.

        Parameters
        -----------
        data : data_iterable
            The data to fit to the distribution too.
            Can be a pd.DataFrame, pd.Series, np.ndarray or any other iterable
            containing data. If not provided, params must be specified.
        params: tuple
            The parameters which define the univariate model.
            See scipy.stats for the correct order.
            If not provided, data must be provided.

        See Also
        --------
        scipy.stats

        Returns
        -------
        fdist: Union[FittedContinuousUnivariate, FittedDiscreteUnivariate]
            A fitted parametric distribution.
        """
        if data is not None:
            try:
                return self._fit_given_data(data, params)
            except Exception as e:
                raise FitError(f"Unable to fit {self.name} distribution to "
                               f"data")

        elif params is not None:
            return self._fit_given_params(params)
        raise ValueError("data and/or params must be given in order to fit "
                         "distribution.")

    @property
    def name(self) -> str:
        """The name of the distribution."""
        return self._name

    @property
    def continuous_or_discrete(self) -> str:
        """returns 'continuous' or 'discrete'."""
        if self.X_DATA_TYPE == float:
            return 'continuous'
        elif self.X_DATA_TYPE == int:
            return 'discrete'
        raise ValueError("X_DATA_TYPE incorrectly set")


class PreFitNumericalUnivariateBase(PreFitUnivariateBase):
    """Base class used to fit a numerical univariate distribution."""
    _PARAMETRIC = 'Non-Parametric/Numerical'

    def __init__(self, name: str, fit: Callable):
        """Base class for fitting or interacting with a numerical /
        non-parametric probability distribution.

        Parameters
        ---------
        name: str
            The name of your univariate distribution.
        fit:
            A callable function which fits data to your distribution.
            Must take a (nx1) numpy array 'data' of sample values, returning a
            tuple of (pdf, cdf, ppf, rvs) where each element of the tuple
            (excluding rvs) is a callable function.
            rvs can be None, in which case it is implemented using inverse
            transform sampling.

        See Also
        --------
        PreFitUnivariateBase
        """
        # argument checks
        if not isinstance(name, str):
            raise TypeError("name must be a string")

        if not callable(fit):
            raise TypeError("Invalid parameter argument in pre-fit "
                            "distribution initialisation.")

        self._name: str = name
        self._fit: Callable = fit
        self.__init()

    def __init(self):
        """resets attributes"""
        self._pdf: Callable = None
        self._cdf: Callable = None
        self._ppf: Callable = None
        self._rvs: Callable = None
        self._support: tuple = None

    def pdf(self, x: Union[float, int, np.ndarray], params: tuple = ()
            ) -> np.ndarray:
        """Not implemented for non-fitted numerical distributions."""
        if self._pdf is None:
            raise NotImplementedError("pdf not implemented for non-fitted "
                                      "numerical distributions.")
        return PreFitUnivariateBase.pdf(self, x, params)

    def cdf(self, x: Union[float, int, np.ndarray], params: tuple = ()
            ) -> np.ndarray:
        """Not implemented for non-fitted numerical distributions."""
        if self._cdf is None:
            raise NotImplementedError("cdf not implemented for non-fitted "
                                      "numerical distributions.")
        return PreFitUnivariateBase.cdf(self, x, params)

    def cdf_approx(self, x: Union[float, int, np.ndarray], params: tuple = (),
                   num_points: int = 100, **kwargs) -> np.ndarray:
        """Not implemented for non-fitted numerical distributions."""
        if self._cdf is None:
            raise NotImplementedError("cdf_approx not implemented for "
                                      "non-fitted numerical distributions.")
        return PreFitUnivariateBase.cdf(self, x, params)

    def ppf(self, q: Union[float, int, np.ndarray], params: tuple = (),
            **kwargs) -> np.ndarray:
        """Not implemented for non-fitted numerical distributions."""
        if self._ppf is None:
            raise NotImplementedError("ppf not implemented for non-fitted "
                                      "numerical distributions.")
        return PreFitUnivariateBase.ppf(self, q, params, **kwargs)

    def ppf_approx(self, q: Union[float, int, np.ndarray], params: tuple,
                   num_points: int = 100, eps: float = 0.01, **kwargs
                   ) -> np.ndarray:
        """Not implemented for non-fitted numerical distributions."""
        if self._ppf is None:
            raise NotImplementedError("ppf_approx not implemented for non-fitted "
                                      "numerical distributions.")
        return PreFitUnivariateBase.ppf(self, q, params, **kwargs)

    def support(self, params: tuple = ()) -> tuple:
        """Not implemented for non-fitted numerical distributions."""
        if self._support is None:
            raise NotImplementedError("support not implemented for non-fitted "
                                      "numerical distributions.")
        return PreFitUnivariateBase.support(self, params)

    def rvs(self, size: tuple, params: tuple = (), **kwargs) -> np.ndarray:
        """Not implemented for non-fitted numerical distributions."""
        if self._rvs is None:
            raise NotImplementedError("rvs not implemented for non-fitted "
                                      "numerical distributions.")
        return PreFitUnivariateBase.rvs(self, size, params)

    def logpdf(self, x: Union[float, int, np.ndarray], params: tuple = ()
               ) -> np.ndarray:
        """Not implemented for non-fitted numerical distributions."""
        if self._pdf is None:
            raise NotImplementedError("logpdf not implemented for non-fitted "
                                      "numerical distributions.")
        return PreFitUnivariateBase.logpdf(self, x, params)

    def likelihood(self, data: np.ndarray, params: tuple = ()) -> float:
        """Not implemented for non-fitted numerical distributions."""
        if self._pdf is None:
            raise NotImplementedError("likelihood not implemented for "
                                      "non-fitted numerical distributions.")
        return PreFitUnivariateBase.likelihood(self, data, params)

    def loglikelihood(self, data: np.ndarray, params: tuple = ()) -> float:
        """Not implemented for non-fitted numerical distributions."""
        if self._pdf is None:
            raise NotImplementedError("loglikelihood not implemented for "
                                      "non-fitted numerical distributions.")
        return PreFitUnivariateBase.loglikelihood(self, data, params)

    def aic(self, data: np.ndarray, params: tuple = ()) -> float:
        """Not implemented for non-fitted numerical distributions."""
        if self._pdf is None:
            raise NotImplementedError("aic not implemented for non-fitted "
                                      "numerical distributions.")
        return PreFitUnivariateBase.aic(self, data, params)

    def bic(self, data: np.ndarray, params: tuple = ()) -> float:
        """Not implemented for non-fitted numerical distributions."""
        if self._pdf is None:
            raise NotImplementedError("bic not implemented for non-fitted "
                                      "numerical distributions.")
        return PreFitUnivariateBase.bic(self, data, params)

    def sse(self, data: np.ndarray, params: tuple = ()) -> float:
        """Not implemented for non-fitted numerical distributions."""
        if self._pdf is None:
            raise NotImplementedError("sse not implemented for non-fitted "
                                      "numerical distributions.")
        return PreFitUnivariateBase.sse(self, data, params)

    def gof(self, data, params: tuple = ()) -> pd.DataFrame:
        """Not implemented for non-fitted numerical distributions."""
        if self._cdf is None:
            raise NotImplementedError("gof not implemented for non-fitted "
                                      "numerical distributions.")
        return PreFitUnivariateBase.gof(self, data, params)

    def plot(self, *args, **kwargs) -> None:
        """Not implemented for non-fitted numerical distributions."""
        raise NotImplementedError("plot not implemented for non-fitted "
                                  "numerical distributions.")

    def fit(self, data: np.ndarray) -> Union[FittedContinuousUnivariate,
                                             FittedDiscreteUnivariate]:
        """Used to fit a numerical univariate distribution to data.

        Parameters
        ---------
        data : data_iterable
            The data to fit to the distribution too.
            Can be a pd.DataFrame, pd.Series, np.ndarray or any other iterable
            containing data.

        Returns
        -------
        fdist: Union[FittedContinuousUnivariate, FittedDiscreteUnivariate]
            A fitted numerical distribution.
        """
        data = check_univariate_data(data)
        params: tuple = ()
        fitted_domain: tuple = (data.min(), data.max())

        # fitting numerical distribution
        (self._pdf, self._cdf, self._ppf, self._support, rvs) = self._fit(data)
        if rvs is None:
            rvs = partial(inverse_transform, ppf=self.ppf)
        self._rvs = rvs

        # fitting empirical distribution
        empirical_fit: Callable = self._get_empirical_fit()
        empirical_pdf, empirical_cdf, empirical_ppf, _, _ = empirical_fit(data)
        empirical_pdf_values: np.ndarray = empirical_pdf(data)

        # fit statistics
        (gof, likelihood, loglikelihood, num_data_points, num_params, aic,
         bic, sse) = self._calc_fit_stats(
            data, params, empirical_pdf_values)

        # returning fitted distribution
        fit_info: dict = {
            "fitted_to_data": True,
            "params": params,
            "support": self.support(),
            "fitted_domain": fitted_domain,
            "empirical_pdf": empirical_pdf,
            "empirical_cdf": empirical_cdf,
            "empirical_ppf": empirical_ppf,
            "gof": gof,
            "likelihood": likelihood,
            "loglikelihood": loglikelihood,
            "num_data_points": num_data_points,
            "num_params": num_params,
            "aic": aic,
            "bic": bic,
            "sse": sse,
            "parametric": self._PARAMETRIC,
        }

        obj = self._FIT_TO(copy.deepcopy(self), fit_info)
        self.__init()
        return obj


class PreFitParametricContinuousUnivariate(PreFitUnivariateBase):
    """A parametric, continuous univariate distribution."""
    _FIT_TO = FittedContinuousUnivariate
    X_DATA_TYPE = float
    _PARAMETRIC = 'Parametric'

    def __init__(self, name: str, pdf: Callable, cdf: Callable, ppf: Callable,
                 support: Callable, fit: Callable, rvs: Callable = None):
        super().__init__(name, pdf, cdf, ppf, support, fit, rvs)
        self._gof: Callable = partial(continuous_gof, cdf=self.cdf,
                                      name=self.name)


class PreFitParametricDiscreteUnivariate(PreFitUnivariateBase):
    """A parametric, discrete univariate distribution."""
    _FIT_TO = FittedDiscreteUnivariate
    X_DATA_TYPE = int
    _PARAMETRIC = 'Parametric'

    def __init__(self, name: str, pdf: Callable, cdf: Callable, ppf: Callable,
                 support: Callable, fit: Callable, rvs: Callable = None):
        super().__init__(name, pdf, cdf, ppf, support, fit, rvs)
        self._gof: Callable = partial(
            discrete_gof, support=self.support, pdf=self.pdf,
            ppf=self.ppf, name=self.name)


class PreFitNumericalContinuousUnivariate(PreFitNumericalUnivariateBase):
    """A numerical, continuous univariate distribution."""
    _FIT_TO = FittedContinuousUnivariate
    X_DATA_TYPE = float

    def _gof(self, data: np.ndarray, params: tuple = ()):
        return continuous_gof(data, params, self.cdf, self.name)


class PreFitNumericalDiscreteUnivariate(PreFitNumericalUnivariateBase):
    """A numerical, discrete univariate distribution."""
    _FIT_TO = FittedDiscreteUnivariate
    X_DATA_TYPE = int

    def _gof(self, data: np.ndarray, params: tuple = ()):
        return discrete_gof(data, params, self.support, self.pdf,
                            self.ppf, self.name)

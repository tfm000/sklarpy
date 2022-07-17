# Contains classes for distributions before and after they are fitted to data
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import logging
from typing import Union
import dill
import os

from sklarpy._utils import prob_bounds, SaveError


__all__ = ['PreFitParametricContinuousUnivariate', 'PreFitParametricDiscreteUnivariate', 'PreFitNumericalContinuousUnivariate', 'PreFitNumericalDiscreteUnivariate']


########################################################################################################################
# Fitted
########################################################################################################################


class FittedUnivariate:
    """Base class for holding a fitted probability distribution."""
    _DIST_TYPE: str

    def __init__(self, obj, fit_info: dict, params: tuple = None):
        """Class used to hold a fitted probability distribution.

        Parameters
        ===========
        obj:
            Either a PreFitContinuousUnivariate or PreFitDiscreteUnivariate object.
        fit_info: dict
            A dictionary containing information about the fit. Contains histogram, fitted domain,
            goodness of fit, sum of squared error, log-likelihood, aic and bic data.
        params: tuple
            The parameters of your fitted distribution. None for a non-parametric distribution.
        """
        self.__obj = obj
        self._fit_info: dict = fit_info
        if params is None:
            params = ()
        self._params: tuple = params
        self.__support = obj.support(*params)
        self.__name_w_params: str = f"{self.name}{tuple(round(p, 2) for p in self._params)}"

    def __str__(self) -> str:
        return self.__name_w_params

    def pdf(self, x: Union[np.ndarray, float, int]) -> np.ndarray:
        """The probability density/mass function.

        Parameters
        ==========
        x: Union[np.ndarray, float, int]
            The values/values to calculate P(X=x) of.

        Returns
        =======
        pdf: np.ndarray
            An array of pdf values
        """
        return self.__obj.pdf(x, *self._params)

    def cdf(self, x: Union[np.ndarray, float, int]) -> np.ndarray:
        """The cumulative density function.

        Parameters
        ==========
        x: Union[np.ndarray, float, int]
            The values/values to calculate P(X<=x) of.

        Returns
        =======
        cdf: np.ndarray
            An array of cdf values
        """
        return self.__obj.cdf(x, *self._params)

    def ppf(self, q: Union[np.ndarray, float]) -> np.ndarray:
        """The cumulative inverse function.

        Parameters
        ==========
        q: Union[np.ndarray, float]
            The value/values to calculate the cdf^-1(q) of. I.e. quantiles.

        Returns
        =======
        ppf: np.ndarray
            An array of quantile values.
        """
        return self.__obj.ppf(q, *self._params)

    def rvs(self, size: tuple) -> np.ndarray:
        """Random sampler.

        Parameters
        ==========
        size: tuple
            The dimensions of the output array you want.

        Returns
        =======
        rvs: np.ndarray
            A random sample of dimension 'size'.
        """
        return self.__obj.rvs(size, *self._params)

    def plot(self, include_sample: bool = True, hist: bool = True, dist_color: str = 'black',
             sample_color: str = 'royalblue', sample_alpha: float=1.0, figsize: tuple = (16, 8), grid: bool = True,
             num: int = 100) -> None:
        """Plots the fitted distribution. Produces subplots of the pdf, cdf, inverse cdf and QQ-plot.

        Parameters
        ==========
        include_sample: bool
            Whether to include the sample/empirical distribution in your plots. Must be true to produce QQ-plots.
            Default is True.
        hist: bool
            True to use a histogram to represent the sample/empirical distribution of the random sample. False to plot
            a line. Default is True.
        dist_color: str
            The color of the fitted distribution to use when plotting. Any acceptable value for the matplotlib.pyplot
            'color' argument can be given. Default is 'black'.
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
        xmin, xmax = self.fitted_domain
        if self.dist_type == 'discrete':
            xrange = range(int(xmin), int(xmax) + 1)
        else:
            xrange = np.linspace(xmin, xmax, num)
        qrange: np.ndarray = np.linspace(*prob_bounds, num)

        # Getting our distribution values
        pdf: np.ndarray = self.pdf(xrange)
        cdf: np.ndarray = self.cdf(xrange)
        ppf: np.ndarray = self.ppf(qrange)

        # Plotting our data
        if include_sample:
            fig, ax = plt.subplots(1, 4, figsize=figsize)
        else:
            fig, ax = plt.subplots(1, 3, figsize=figsize)
        if hist:
            sample_plot = (lambda idx, rng, val, c, a, lab: ax[idx].bar(rng, val, color=c, alpha=a, label=lab))
        else:
            sample_plot = (lambda idx, rng, val, c, a, lab: ax[idx].plot(rng, val, color=c, alpha=a, label=lab))

        # Plotting fitted distribution
        ax[0].plot(xrange, pdf, label=self.__name_w_params, c=dist_color)
        ax[1].plot(xrange, cdf, label=self.__name_w_params, c=dist_color)
        ax[2].plot(qrange, ppf, label=self.__name_w_params, c=dist_color)

        if include_sample:
            # Getting our sample/empirical distribution values
            sample_range, sample_pdf, sample_cdf = self._fit_info['histogram']
            qq: np.ndarray = self.ppf(sample_cdf)

            # Plotting sample
            sample_plot(0, sample_range, sample_pdf, sample_color, sample_alpha, 'Sample')
            sample_plot(1, sample_range, sample_cdf, sample_color, sample_alpha, 'Sample')
            ax[2].plot(sample_cdf, sample_range, c=sample_color, label='Sample')
            ax[3].plot(xrange, xrange, c='black', label='y=x')
            ax[3].scatter(qq[:-1], sample_range[:-1], label='quartiles')
            ax[3].set_title('QQ-Plot')
            ax[3].set_xlabel("Theoretical Quantiles")
            ax[3].set_ylabel("Sample Quantiles")
            ax[3].grid(grid)
            ax[3].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=1)

        xlabels = ("x", "x", "P(X<=q)")
        ylabels = ("P(X=x)", "P(X<=x)", "q")
        titles = ('PDF', 'CDF', 'Inverse CDF')
        for i in range(3):
            ax[i].set_xlabel(xlabels[i])
            ax[i].set_ylabel(ylabels[i])
            ax[i].set_title(titles[i])
            ax[i].grid(grid)
            ax[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=1)
        plt.tight_layout()
        plt.show()

    def gof(self, data: np.ndarray = None) -> pd.Series:
        """Calculates goodness of fit tests for the specified distribution against a random sample.

        Tests performed
        ----------------
        Cramér-von Mises:
            One-sample Cramér-von Mises goodness of fit test.
            Tests the null hypothesis that the random sample has a cumulative distribution function specified by
            your distribution.
        Kolmogorov-Smirnov:
            Two sample Kolmogorov-Smirnov goodness of fit test.
            Tests the null hypothesis that the random sample has a cumulative distribution function specified by
            your distribution.

        Parameters
        ==========
        data: np.ndarray
            A (nx1) numpy array containing the random sample. If not given, the data used to fit the distribution is
            used.

        Returns
        =======
        res: pd.Series
            A pandas series containing the test statistics and p-values of your gof tests.
        """
        if data is None:
            return self._fit_info['gof']
        return self.__obj.gof(data, *self._params)

    def save(self, file: Union[str, None] = None, fix_extension: bool = True, overwrite_existing: bool = True):
        """Saves univariate distribution as a pickled file.

        Parameters
        ==========
        file: Union[str, None]
            The file to save. If None, the distribution is saved under the distribution's name in the current working
            directory. If a file is given, it must include the full file path. The .pickle extension is optional
            provided fix_extension is True.
        fix_extension: bool
            Whether to replace any existing extension with the '.pickle' file extension. Default is True.
        overwrite_existing: bool
            True to overwrite existing files saved under the same name. False to save under a unique name.
            Default is True.

        See Also
        ---------
        sklarpy.load
        pickle
        dill
        """
        # Input checks
        if file is None:
            dir_path: str = os.getcwd()
            file = f'{dir_path}\\{self.name}'
        if not isinstance(file, str):
            raise TypeError("file argument must be a string.")
        if not isinstance(fix_extension, bool):
            raise TypeError("fix_extension argument must be a boolean.")

        # Changing file extension to .pickle
        file_name, extension = os.path.splitext(file)
        if fix_extension:
            extension = '.pickle'

        if not overwrite_existing:
            # Saving under a unique file name
            count: int = 0
            unique_str: str = ''
            while os.path.exists(f'{file_name}{unique_str}{extension}'):
                count += 1
                unique_str = f'({count})'
            file_name = f'{file_name}{unique_str}'

        try:
            with open(f'{file_name}{extension}', 'wb') as f:
                dill.dump(self, f)
        except Exception as e:
            raise SaveError(e)

    @property
    def summary(self) -> pd.Series:
        """Summary of the fitted distribution."""
        s1: pd.Series = pd.Series(
            [self.__obj._PARAMETRIC, self._DIST_TYPE, self.name, *self._params, self.support, self._fit_info['fitted_domain']],
            index=['Parametric/Non-Parametric', 'Discrete/Continuous', 'Distribution',  *[f"param{i}" for i in range(len(self.params))],
                   'Distribution Support', 'Fitted Domain'])
        s2: pd.Series = self.gof()
        s3: pd.Series = pd.Series(
            [self.loglikelihood, self.aic, self.bic, self.sse, self._fit_info['N']],
            index=['Log-likelihood', 'AIC', 'BIC', 'Sum of Squared Error', 'Number of Data Points'])

        summary: pd.Series = pd.concat([s1, s2, s3], axis=0)
        summary.name = "summary"
        return summary

    @property
    def loglikelihood(self) -> float:
        """The log-likelihood of the fitted distribution."""
        return self._fit_info['loglikelihood']

    @property
    def aic(self) -> float:
        """The Akaike Information Criterion (AIC) of the fitted distribution."""
        return self._fit_info['aic']

    @property
    def bic(self) -> float:
        """The Bayesian Information Criterion (BIC) of the fitted distribution."""
        return self._fit_info['bic']

    @property
    def sse(self) -> float:
        """The sum of squared error between the empirical and fitted distribution's pdf/pmf values."""
        return self._fit_info['sse']

    @property
    def name(self) -> str:
        """The name of the distribution."""
        return self.__obj.name

    @property
    def name_with_params(self) -> str:
        """The name of the distribution with its fitted parameters."""
        return self.__name_w_params

    @property
    def params(self) -> tuple:
        """The parameters of your fitted distribution."""
        return self._params

    @property
    def support(self) -> tuple:
        """The support of your distribution."""
        return self.__support

    @property
    def fitted_domain(self) -> tuple:
        """The fitted domain of your distribution"""
        return self._fit_info["fitted_domain"]

    @property
    def dist_type(self) -> str:
        """The type of distribution."""
        return self._DIST_TYPE


class FittedContinuousUnivariate(FittedUnivariate):
    """Holds a fitted continuous probability distribution."""
    _DIST_TYPE = "continuous"

    def gof(self, data: np.ndarray = None) -> pd.Series:
        """Calculates goodness of fit tests for the specified distribution against a random sample. If no data is given
        the gof of the distribution to its fitted data is performed.

        Continuous tests performed
        ----------------
        Cramér-von Mises:
            One-sample Cramér-von Mises goodness of fit test.
            Tests the null hypothesis that the random sample has a cumulative distribution function specified by
            your distribution.
        Kolmogorov-Smirnov:
            Two sample Kolmogorov-Smirnov goodness of fit test.
            Tests the null hypothesis that the random sample has a cumulative distribution function specified by
            your distribution.

        Discrete tests performed
        ----------------
        chi-squared:
            Tests the null hypothesis that the observed discrete data has frequencies given by the specified
            distribution.

        Parameters
        ==========
        data: np.ndarray
            A (nx1) numpy array containing the random sample. If not given, the data the distribution was fitted to is
            used.

        Returns
        =======
        res: pd.Series
            A pandas series containing the test statistics and p-values of your gof tests.
        """
        return super().gof(data)


class FittedDiscreteUnivariate(FittedUnivariate):
    """Holds a fitted discrete probability distribution."""
    _DIST_TYPE = "discrete"

    def gof(self, data: np.ndarray = None) -> pd.Series:
        """Calculates goodness of fit tests for the specified distribution against a random sample.

        Tests performed
        ----------------
        chi-squared:
            Tests the null hypothesis that the observed discrete data has frequencies given by the specified
            distribution.

        Parameters
        ==========
        data: np.ndarray
            A (nx1) numpy array containing the random sample.

        Returns
        =======
        res: pd.Series
            A pandas series containing the test statistics and p-values of your gof tests.
        """
        return super().gof(data)


########################################################################################################################
# Pre-Fitted
########################################################################################################################


class PreFitContinuousUnivariate:
    """Base class for fitting or interacting with a continuous probability distribution."""
    _FIT_TO = FittedContinuousUnivariate
    _DIST_TYPE = 'continuous'

    @staticmethod
    def _empirical(data: np.ndarray) -> tuple:
        """Creates an empirical pdf and cdf from the data.

        Parameters
        ==========
        data: np.ndarray
            The random sample from which to construct your empirical pdf and cdf.

        Returns
        =======
        empirical: tuple
            empirical_range, empirical_pdf, empirical_cdf, fitted_domain
        """
        xmin, xmax, N = data.min(), data.max(), len(data)
        num_bins: int = min(len(set(data)), 100)  # number of bins to group our continuous data into
        empirical_pdf, empirical_range = np.histogram(data, bins=num_bins)
        empirical_range = (empirical_range[1:] + empirical_range[:-1]) / 2
        empirical_cdf = np.cumsum(empirical_pdf)
        bin_width = (xmax - xmin) / num_bins
        empirical_cdf, empirical_pdf = empirical_cdf / N, empirical_pdf / (N * bin_width)
        return empirical_range, empirical_pdf, empirical_cdf, (xmin, xmax)

    def gof(self, data: np.ndarray, *params) -> pd.Series:
        """Calculates goodness of fit tests for the specified distribution against a random sample.

        Tests performed
        ----------------
        Cramér-von Mises:
            One-sample Cramér-von Mises goodness of fit test.
            Tests the null hypothesis that the random sample has a cumulative distribution function specified by
            your distribution.
        Kolmogorov-Smirnov:
            Two sample Kolmogorov-Smirnov goodness of fit test.
            Tests the null hypothesis that the random sample has a cumulative distribution function specified by
            your distribution.

        Parameters
        ==========
        data: np.ndarray
            A (nx1) numpy array containing the random sample.
        params:
            The parameters specifying the distribution.

        Returns
        =======
        res: pd.Series
            A pandas series containing the test statistics and p-values of your gof tests.
        """
        data = np.asarray(data).flatten()
        cdf = (lambda x: self.cdf(x, *params))

        # Cramér-von Mises gof test
        cvm_res = scipy.stats.cramervonmises(data, cdf)

        # Kolmogorov-Smirnov gof test
        ks_stat, ks_pvalue = scipy.stats.kstest(data, cdf)

        values = [cvm_res.statistic, cvm_res.pvalue, ks_stat, ks_pvalue]
        index = ['Cramér-von Mises statistic', 'Cramér-von Mises p-value',
                 'Kolmogorov-Smirnov statistic', 'Kolmogorov-Smirnov p-value']
        return pd.Series(values, index=index, name=self.name)

    def fit(self, data: np.ndarray) -> FittedContinuousUnivariate:
        return super().fit(data)


class PreFitDiscreteUnivariate:
    """Base class for fitting or interacting with a discrete probability distribution."""
    _FIT_TO = FittedDiscreteUnivariate
    _DIST_TYPE = 'discrete'

    @staticmethod
    def _empirical(data: np.ndarray) -> tuple:
        """Creates an empirical pdf and cdf from the data.

        Parameters
        ==========
        data: np.ndarray
            The random sample from which to construct your empirical pdf and cdf.

        Returns
        =======
        empirical: tuple
            empirical_range, empirical_pdf, empirical_cdf, fitted_domain
        """
        xmin, xmax, N = data.min(), data.max(), len(data)
        empirical_range: np.ndarray = np.arange(xmin, xmax + 1)
        empirical_pdf: np.ndarray = np.array([np.count_nonzero(data == i)/N for i in empirical_range])
        empirical_cdf = np.cumsum(empirical_pdf)
        return empirical_range, empirical_pdf, empirical_cdf, (xmin, xmax)

    def gof(self, data: np.ndarray, *params) -> pd.Series:
        """Calculates goodness of fit tests for the specified distribution against a random sample.

        Tests performed
        ----------------
        chi-squared:
            Tests the null hypothesis that the observed discrete data has frequencies given by the specified
            distribution.

        Parameters
        ==========
        data: np.ndarray
            A (nx1) numpy array containing the random sample.
        params:
            The parameters specifying the distribution.

        Returns
        =======
        res: pd.Series
            A pandas series containing the test statistics and p-values of your gof tests.
        """
        data = np.asarray(data).flatten()
        num = len(data)
        dof = num - len(params) - 1

        support: list = list(self.support(*params))
        if abs(support[0]) == np.inf:
            support[0] = int(self.ppf(prob_bounds[0], *params))
        if abs(support[1]) == np.inf:
            support[1] = int(self.ppf(prob_bounds[1], *params))

        # Chi-squared gof test
        xrange = range(int(support[0]), int(support[1]) + 1)
        observed = np.array([np.count_nonzero(data == x) for x in xrange])
        expected = self.pdf(xrange, *params) * num
        index = np.where(expected != 0)[0]
        expected = expected[index]
        observed = observed[index]
        chisq_stat = np.sum(((expected - observed) ** 2) / expected)
        chisq_pvalue = scipy.stats.chi2.sf(chisq_stat, dof)
        values = [float(chisq_stat), float(chisq_pvalue)]
        index = ['chi-square statistic', 'chi-square p-value']
        return pd.Series(values, index=index, name=self.name)

    def fit(self, data: np.ndarray) -> FittedDiscreteUnivariate:
        return super().fit(data)


class PreFitParametricUnivariate:
    """Base class for fitting or interacting with a parametric probability distribution."""
    _FIT_TO: FittedUnivariate
    _PARAMETRIC: str = 'Parametric'
    _DIST_TYPE: str

    def __init__(self, name: str, pdf, cdf, ppf, support, fit, rvs=None):
        """Base class for fitting or interacting with a parametric probability distribution.

        Parameters
        ==========
        name: str
            The name of your univariate distribution.
        pdf:
            A callable function of the pdf/pmf of your univariate distribution. Must take a (nx1) numpy array 'x' of
            domain values and parameters specifying your distribution as arguments, returning a (nx1) numpy array of
            P(X=x) probabilities.
        cdf:
            A callable function of the cdf of your univariate distribution. Must take a (nx1) numpy array 'x' of
            domain values and parameters specifying your distribution as arguments, returning a (nx1) numpy array of
            P(X<=x) probabilities.
        ppf:
            A callable function of the cdf inverse/quantile function of your univariate distribution. Must take a
            (nx1) numpy array 'q' of cdf values [0, 1] and parameters specifying your distribution as arguments,
            returning a (nx1) numpy array of x = cdf^-1(q) values.
        support:
            A callable function of the support of your univariate distribution. Must take parameters specifying your
            distribution as arguments, returning a tuple of your distribution's domain.
        fit:
            A callable function which fits data to your univariate distribution. Must take a (nx1) numpy array 'x' of
            domain values, returning the parameters specifying your distribution as a tuple.
        rvs:
            A callable function which produces random samples from your univariate distribution. Must take a tuple
            'size' of the number of samples to generate and parameters specifying your distribution as arguments,
            returning a numpy array of dimension 'size' of random samples. If not given, this is implemented using
            inverse transform sampling.

        """
        for func in (pdf, cdf, ppf, support, fit):
            if not callable(func):
                raise TypeError("Invalid parameter argument in pre-fit distribution initialisation.")

        self.name: str = name
        self._pdf = pdf
        self._cdf = cdf
        self._ppf = ppf
        self.support = support
        self.__fit = fit
        if (rvs is None) or callable(rvs):
            self._rvs = rvs
        else:
            raise TypeError("Invalid parameter argument in pre-fit distribution initialisation.")

    def __str__(self) -> str:
        return f"PreFit{self.name.title()}Distribution"

    def pdf(self, x: Union[np.ndarray, float, int], *params) -> np.ndarray:
        """The probability density/mass function.

        Parameters
        ==========
        x: Union[np.ndarray, float, int]
            The values/values to calculate P(X=x) of.
        params:
            The parameters specifying the distribution.

        Returns
        ========
        pdf: np.ndarray
            An array of pdf values.
        """
        support: tuple = self.support(*params)
        vals: np.ndarray = self._pdf(np.asarray(x).flatten(), *params)
        vals = np.where(x >= support[0], vals, 0)
        vals = np.where(x <= support[1], vals, 0)
        return vals

    def cdf(self, x: Union[np.ndarray, float, int], *params) -> np.ndarray:
        """The cumulative density function.

        Parameters
        ==========
        x: Union[np.ndarray, float, int]
            The values/values to calculate P(X<=x) of.
        params:
            The parameters specifying the distribution.

        Returns
        =======
        cdf: np.ndarray
            An array of cdf values.
        """
        support: tuple = self.support(*params)
        vals: np.ndarray = self._cdf(np.asarray(x).flatten(), *params)
        vals = np.where(x >= support[0], vals, 0)
        vals = np.where(x <= support[1], vals, 1)
        return vals

    def ppf(self, q: Union[np.ndarray, float], *params) -> np.ndarray:
        """The cumulative inverse function.

        Parameters
        ==========
        q: Union[np.ndarray, float]
            The value/values to calculate the cdf^-1(q) of. I.e. quantiles.
        params:
            The parameters specifying the distribution.

        Returns
        =======
        ppf: np.ndarray
            An array of quantile values.
        """
        q = np.asarray(q).flatten()
        q = np.where(q != 0, q, prob_bounds[0])
        q = np.where(q != 1, q, prob_bounds[1])
        return self._ppf(np.asarray(q).flatten(), *params)

    def rvs(self, size: tuple, *params) -> np.ndarray:
        """Random sampler.

        Parameters
        ==========
        size: tuple
            The dimensions of the output array you want.
        params:
            The parameters specifying the distribution.

        Returns
        =======
        rvs: np.ndarray
            A random sample of dimension 'size'.
        """
        if self._rvs is not None:
            return self._rvs(*params, size=size)

        # rvs not given in init - generating random data
        u = np.random.uniform(size=size)
        return self.ppf(u, *params)

    def fit(self, data: np.ndarray) -> Union[FittedContinuousUnivariate, FittedDiscreteUnivariate]:
        """Fits the distribution to the data.

        Parameters
        ==========
        data: np.ndarray
            A (nx1) numpy array containing the random sample to fit the distribution to.

        Returns
        =======
        fitted: Union[FittedContinuousUnivariate, FittedDiscreteUnivariate]
            A fitted univariate probability distribution.
        """
        data = np.asarray(data).flatten()
        params: tuple = self.__fit(data)
        N: int = len(data)
        k: int = len(params)

        # Getting infomation about the empirical distribution
        empirical_range, empirical_pdf, empirical_cdf, fitted_domain = self._empirical(data)

        # GoF info
        gof: pd.Series = self.gof(data, *params)

        # log-likelihood, aic, bic ans sse
        pdf_values = np.clip(self.pdf(empirical_range, *params), *prob_bounds)  # clipping to prevent log-likelihood being inf at prob = 0
        loglikelihood = np.sum(np.log(pdf_values))
        aic = 2 * k - 2 * loglikelihood
        bic = -2 * loglikelihood + np.log(N) * k
        sse = np.sum((pdf_values - empirical_pdf) ** 2)

        fit_info: dict = {
            "histogram": (empirical_range, empirical_pdf, empirical_cdf),
            "fitted_domain": fitted_domain,
            "gof": gof,
            "loglikelihood": loglikelihood,
            "aic": aic,
            "bic": bic,
            "sse": sse,
            "N": N,
        }

        return self._FIT_TO(self, fit_info, params)

    @property
    def dist_type(self) -> str:
        """Whether the distribution is continuous or discrete."""
        return self._DIST_TYPE

class PreFitParametricContinuousUnivariate(PreFitContinuousUnivariate, PreFitParametricUnivariate):
    """Class for fitting or interacting with a continuous parametric probability distribution."""
    pass


class PreFitParametricDiscreteUnivariate(PreFitDiscreteUnivariate, PreFitParametricUnivariate):
    """Class for fitting or interacting with a discrete parametric probability distribution."""
    pass


class PreFitNumericalUnivariate(PreFitParametricUnivariate): #this inheritence will be removed, maybe have child PreFitNumericalContinuousUnivariate classes inherite PreFitContinuousUnivariate for gof etc -> wont work in current form as relys on params
    """Base class for fitting or interacting with a numerical/non-parametric probability distribution."""
    _PARAMETRIC = 'Non-Parametric'

    def __init__(self, name: str, fit):
        """Base class for fitting or interacting with a numerical/non-parametric probability distribution.

        Parameters
        ==========
        name: str
            The name of your univariate distribution.
        fit:
            A callable function which fits data to your distribution. Must take a (nx1) numpy array 'x' of domain
            values, returning a tuple of (pdf, cdf, ppf, rvs) where each element of the tuple (excluding rvs) is a
            callable function. rvs can be None, in which case it is implemented using inverse transform sampling.
        """
        self.name: str = name
        if not callable(fit):
            raise TypeError("Invalid parameter argument in pre-fit distribution initialisation.")
        self.__fit = fit

        self._pdf = None
        self._cdf = None
        self._ppf = None
        self._rvs = None

    def pdf(self, x: Union[np.ndarray, float, int]) -> np.ndarray:
        """Not implemented for non-fitted numerical univariate distributions."""
        if self._pdf is None:
            logging.warning("PDF not implemented for non-fitted numerical univariate distributions.")
            return np.full(np.asarray(x).shape, np.nan)
        return PreFitParametricUnivariate.pdf(self, x, *())


    def cdf(self, x: Union[np.ndarray, float, int]) -> np.ndarray:
        """Not implemented for non-fitted numerical univariate distributions."""
        if self._cdf is None:
            logging.warning("CDF not implemented for non-fitted numerical univariate distributions.")
            return np.full(np.asarray(x).shape, np.nan)
        return PreFitParametricUnivariate.cdf(self, x, *())

    def ppf(self, q: Union[np.ndarray, float]) -> np.ndarray:
        """Not implemented for non-fitted numerical univariate distributions."""
        if self._ppf is None:
            logging.warning("PPF not implemented for non-fitted numerical univariate distributions.")
            return np.full(np.asarray(q).shape, np.nan)
        return PreFitParametricUnivariate.ppf(self, q, *())

    def rvs(self, size: tuple) -> np.ndarray:
        """Not implemented for non-fitted numerical univariate distributions."""
        if self._pdf is None:
            logging.warning("rvs not implemented for non-fitted numerical univariate distributions.")
            return np.full(size, np.nan)
        return PreFitParametricUnivariate.rvs(self, size, *())

    def gof(self, data: np.ndarray) -> pd.Series:
        """Not implemented for non-fitted numerical univariate distributions."""
        if self._pdf is None:
            logging.warning("gof not implemented for non-fitted numerical univariate distributions.")
            return pd.Series([])
        return super().gof(data)

    def fit(self, data: np.ndarray):
        data = np.asarray(data).flatten()
        N: int = len(data)
        k: int = 0
        pdf, cdf, ppf, support, rvs = self.__fit(data)

        for func in (pdf, cdf, ppf):
            if not callable(func):
                raise TypeError("Invalid parameter argument in pre-fit distribution initialisation.")

        self._pdf = pdf
        self._cdf = cdf
        self._ppf = ppf
        self.support = (lambda: support)
        self._rvs = rvs

        # Getting infomation about the empirical distribution
        empirical_range, empirical_pdf, empirical_cdf, fitted_domain = self._empirical(data)

        # GoF info
        gof: pd.Series = self.gof(data)

        # log-likelihood, aic, bic ans sse
        pdf_values = np.clip(self.pdf(empirical_range), *prob_bounds)  # clipping to prevent log-likelihood being inf at prob = 0
        loglikelihood = np.sum(np.log(pdf_values))
        aic = 2 * k - 2 * loglikelihood
        bic = -2 * loglikelihood + np.log(N) * k
        sse = np.sum((pdf_values - empirical_pdf) ** 2)

        fit_info: dict = {
            "histogram": (empirical_range, empirical_pdf, empirical_cdf),
            "fitted_domain": fitted_domain,
            "gof": gof,
            "loglikelihood": loglikelihood,
            "aic": aic,
            "bic": bic,
            "sse": sse,
            "N": N,
        }

        return self._FIT_TO(self, fit_info, ())


class PreFitNumericalContinuousUnivariate(PreFitNumericalUnivariate, PreFitContinuousUnivariate):
    """Class for fitting or interacting with a continuous numerical/non-parametric probability distribution."""
    pass


class PreFitNumericalDiscreteUnivariate(PreFitDiscreteUnivariate, PreFitNumericalUnivariate):
    """Class for fitting or interacting with a discrete numerical/non-parametric probability distribution."""
    pass

from typing import Callable, Union
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklarpy.algorithms import inverse_transform, continuous_goodness_of_fit, discrete_goodness_of_fit
from sklarpy._utils import num_or_array, univariate_num_to_array, check_params, check_univariate_data, FitError
from sklarpy.univariate._fitted_dists import FittedContinuousUnivariate, FittedDiscreteUnivariate
from sklarpy.univariate._fit import discrete_empirical_fit, continuous_empirical_fit, poisson_fit


class PreFitUnivariateBase(object):
    _FIT_TO: Callable
    X_DATA_TYPE = None

    def __init__(self, name: str, pdf: Callable, cdf: Callable, ppf: Callable, support: Callable, fit: Callable, rvs: Callable=None):
        # argument checks
        if not isinstance(name, str):
            raise TypeError("name must be a string")

        for func in (pdf, cdf, ppf, support, fit):
            if not callable(func):
                raise TypeError("Invalid argument in pre-fit distribution initialisation.")

        if rvs is None:
            rvs = partial(inverse_transform, ppf=ppf)
        elif not callable(rvs):
            raise TypeError("Invalid argument in pre-fit distribution initialisation.")

        self._name: str = name
        self._pdf: Callable = pdf
        self._cdf: Callable = cdf
        self._ppf: Callable = ppf
        self._support: Callable = support
        self._fit: Callable = fit
        self._rvs: Callable = rvs
        self._gof: Callable = None

    def __str__(self) -> str:
        return f"PreFit{self._name.title()}Distribution"

    def __repr__(self) -> str:
        return self.__str__()

    def pdf(self, x: num_or_array, params: tuple) -> np.ndarray:
        x: np.ndarray = univariate_num_to_array(x)
        params: tuple = check_params(params)
        return self._pdf(x, *params)

    def cdf(self, x: num_or_array, params: tuple) -> np.ndarray:
        x: np.ndarray = univariate_num_to_array(x)
        params: tuple = check_params(params)
        return self._cdf(x, *params)

    def ppf(self, q: num_or_array, params: tuple) -> np.ndarray:
        q: np.ndarray = univariate_num_to_array(q)
        params: tuple = check_params(params)
        return self._ppf(q, *params)

    def support(self, params: tuple) -> tuple:
        params: tuple = check_params(params)
        return self._support(*params)

    def rvs(self, size: tuple, params: tuple) -> np.ndarray:
        if not isinstance(size, tuple):
            raise TypeError("size must be a tuple.")
        elif len(size) < 1:
            raise ValueError("size must not be empty.")
        params: tuple = check_params(params)
        return self._rvs(*params, size=size)

    def logpdf(self, x: num_or_array, params: tuple) -> np.ndarray:
        return np.log(self.pdf(x, params))

    def likelihood(self, data: np.ndarray, params: tuple) -> float:
        data = check_univariate_data(data)
        return float(np.product(self.pdf(data, params)))

    def loglikelihood(self, data: np.ndarray, params: tuple) -> float:
        data = check_univariate_data(data)
        return float(np.sum(self.logpdf(data, params)))

    def aic(self, data: np.ndarray, params: tuple) -> float:
        loglikelihood: float = self.loglikelihood(data, params)
        return 2 * (len(params) - loglikelihood)

    def bic(self, data: np.ndarray, params: tuple) -> float:
        loglikelihood: float = self.loglikelihood(data, params)
        return -2 * loglikelihood + np.log(data.size) * len(params)

    def sse(self, data: np.ndarray, params: tuple) -> float:
        pdf_values: np.ndarray = self.pdf(data, params)
        empirical_pdf, _, _, _, _ = self._EMPIRICAL_FIT(data)
        empirical_pdf_values: np.ndarray = empirical_pdf(data)
        return float(np.sum((pdf_values - empirical_pdf_values) ** 2))

    def gof(self, data, params: tuple) -> pd.DataFrame:
        data = check_univariate_data(data)
        params = check_params(params)
        return self._gof(data, params)

    def plot(self, params: tuple, data: np.ndarray = None, empirical_hist: bool = True, color: str = 'black', empirical_color: str = 'royalblue', empirical_alpha: float = 1.0, figsize: tuple = (16, 8), grid: bool = True, num_to_plot: int = 100) -> None:
        pass

    def _fit_given_params(self, params: tuple) -> Union[FittedContinuousUnivariate, FittedDiscreteUnivariate]:
        params = check_params(params)
        support: tuple = self.support(params)

        # returning fitted distribution
        fit_info: dict = {
            "fitted_to_data": False,
            "params": params,
            "support": support,
            "fitted_domain": (),
            "gof": pd.DataFrame(),
            "likelihood": np.nan,
            "loglikelihood": np.nan,
            "num_data_points": np.nan,
            "num_params": len(params),
            "aic": np.nan,
            "bic": np.nan,
            "sse": np.nan,
        }
        return self._FIT_TO(self, fit_info)

    def _fit_given_data(self, empirical_fit: Callable, data: np.ndarray, params: tuple = None) -> Union[FittedContinuousUnivariate, FittedDiscreteUnivariate]:
        data = check_univariate_data(data)
        if params is None:
            params: tuple = self._fit(data)
        else:
            params = check_params(params)
        support: tuple = self.support(params)
        fitted_domain: tuple = data.min(), data.max()

        # fitting empirical distribution
        empirical_pdf, empirical_cdf, empirical_ppf, _, _ = empirical_fit(data)
        empirical_pdf_values: np.ndarray = empirical_pdf(data)

        # fit statistics
        gof: pd.DataFrame = self._gof(data, params)
        pdf_values: np.ndarray = self.pdf(data, params)
        likelihood: float = float(np.product(pdf_values))
        loglikelihood: float = float(np.sum(np.log(pdf_values)))
        num_data_points: int = len(data)
        num_params: int = len(params)
        aic: float = 2 * num_params - 2 * loglikelihood
        bic: float = -2 * loglikelihood + np.log(num_data_points) * num_params
        sse: float = float(np.sum((pdf_values - empirical_pdf_values) ** 2))

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
        }
        return self._FIT_TO(self, fit_info)

    def fit(self, empirical_fit: Callable, data: np.ndarray = None, params: tuple = None) -> Union[FittedContinuousUnivariate, FittedDiscreteUnivariate]:
        if data is not None:
            return self._fit_given_data(empirical_fit, data, params)
        elif params is not None:
            return self._fit_given_params(params)
        raise ValueError("data and/or params must be given in order to fit distribution.")

    @property
    def name(self) -> str:
        return self._name


class PreFitContinuousUnivariate(PreFitUnivariateBase):
    _FIT_TO = FittedContinuousUnivariate
    X_DATA_TYPE = float

    def __init__(self, name: str, pdf: Callable, cdf: Callable, ppf: Callable, support: Callable, fit: Callable, rvs: Callable=None):
        PreFitUnivariateBase.__init__(self, name, pdf, cdf, ppf, support, fit, rvs)
        self._gof: Callable = partial(continuous_goodness_of_fit, cdf=self.cdf, name=self.name)

    def fit(self, data: np.ndarray = None, params: tuple = None) -> FittedContinuousUnivariate:
        return PreFitUnivariateBase.fit(self, continuous_empirical_fit, data, params)


class PreFitDiscreteUnivariate(PreFitUnivariateBase):
    _FIT_TO = FittedDiscreteUnivariate
    X_DATA_TYPE = int

    def __init__(self, name: str, pdf: Callable, cdf: Callable, ppf: Callable, support: Callable, fit: Callable, rvs: Callable = None):
        PreFitUnivariateBase.__init__(self, name, pdf, cdf, ppf, support, fit, rvs)
        self._gof: Callable = partial(discrete_goodness_of_fit, support=self.support, pdf=self.pdf, ppf=self.ppf, name=self.name)

    def fit(self, data: np.ndarray = None, params: tuple = None) -> FittedDiscreteUnivariate:
        return PreFitUnivariateBase.fit(self, discrete_empirical_fit, data, params)

###############################################################


class PreFitNumericalUnivariateBase(PreFitUnivariateBase):
    def __init__(self, name: str, fit: Callable):
        """Base class for fitting or interacting with a numerical/non-parametric probability distribution.

        Parameters
        ==========
        name: str
            The name of your univariate distribution.
        fit:
            A callable function which fits data to your distribution. Must take a (nx1) numpy array 'data' of sample
            values, returning a tuple of (pdf, cdf, ppf, rvs) where each element of the tuple (excluding rvs) is a
            callable function. rvs can be None, in which case it is implemented using inverse transform sampling.
        """
        # argument checks
        if not isinstance(name, str):
            raise TypeError("name must be a string")

        if not callable(fit):
            raise TypeError("Invalid parameter argument in pre-fit distribution initialisation.")

        self._name: str = name
        self._fit: Callable = fit
        self._pdf: Callable = None
        self._cdf: Callable = None
        self._ppf: Callable = None
        self._rvs: Callable = None

    def pdf(self, x: num_or_array, params: tuple) -> np.ndarray:
        if self._pdf is None:
            raise FitError("PDF not implemented for non-fitted numerical distributions.")
        return PreFitUnivariateBase.pdf(self, x, params)

    def cdf(self, x: num_or_array, params: tuple) -> np.ndarray:
        if self._cdf is None:
            raise FitError("CDF not implemented for non-fitted numerical distributions.")
        return PreFitUnivariateBase.cdf(self, x, params)

    def ppf(self, q: num_or_array, params: tuple) -> np.ndarray:
        if self._ppf is None:
            raise FitError("PPF not implemented for non-fitted numerical distributions.")
        return PreFitUnivariateBase.ppf(self, q, params)

    def support(self, params: tuple) -> tuple:
        if self._support is None:
            raise FitError("support not implemented for non-fitted numerical distributions.")
        return PreFitUnivariateBase.support(self, params)

    def rvs(self, size: tuple, params: tuple) -> np.ndarray:
        if self._rvs is None:
            raise FitError("rvs not implemented for non-fitted numerical distributions.")
        return PreFitUnivariateBase.rvs(self, size, params)

    def logpdf(self, x: num_or_array, params: tuple) -> np.ndarray:
        if self._pdf is None:
            raise FitError("log-PDF not implemented for non-fitted numerical distributions.")
        return PreFitUnivariateBase.logpdf(self, x, params)

    def likelihood(self, data: np.ndarray, params: tuple) -> float:
        if self._pdf is None:
            raise FitError("likelihood not implemented for non-fitted numerical distributions.")
        return PreFitUnivariateBase.likelihood(self, data, params)

    def loglikelihood(self, data: np.ndarray, params: tuple) -> float:
        if self._pdf is None:
            raise FitError("log-likelihood not implemented for non-fitted numerical distributions.")
        return PreFitUnivariateBase.loglikelihood(self, data, params)

    def aic(self, data: np.ndarray, params: tuple) -> float:
        if self._pdf is None:
            raise FitError("aic not implemented for non-fitted numerical distributions.")
        return PreFitUnivariateBase.aic(self, data, params)

    def bic(self, data: np.ndarray, params: tuple) -> float:
        if self._pdf is None:
            raise FitError("bic not implemented for non-fitted numerical distributions.")
        return PreFitUnivariateBase.bic(self, data, params)

    def sse(self, data: np.ndarray, params: tuple) -> float:
        if self._pdf is None:
            raise FitError("sse not implemented for non-fitted numerical distributions.")
        return PreFitUnivariateBase.sse(self, data, params)

    # def gof(self, data, params: tuple) -> pd.DataFrame:
    #     data = check_univariate_data(data)
    #     params = check_params(params)
    #     return self._gof(data, params)

    def plot(self, params: tuple, data: np.ndarray = None, empirical_hist: bool = True, color: str = 'black', empirical_color: str = 'royalblue', empirical_alpha: float = 1.0, figsize: tuple = (16, 8), grid: bool = True, num_to_plot: int = 100) -> None:
        raise FitError("plot not implemented for non-fitted numerical distributions.")

    def fit(self, data: np.ndarray) -> Union[FittedContinuousUnivariate, FittedDiscreteUnivariate]:
        #(TO)DO this onwards
        data = check_univariate_data(data)
        if params is None:
            params: tuple = self._fit(data)
        else:
            params = check_params(params)
        support: tuple = self.support(params)
        fitted_domain: tuple = data.min(), data.max()

        # fitting empirical distribution
        empirical_pdf, empirical_cdf, empirical_ppf, _, _ = empirical_fit(data)
        empirical_pdf_values: np.ndarray = empirical_pdf(data)

        # fit statistics
        gof: pd.DataFrame = self._gof(data, params)
        pdf_values: np.ndarray = self.pdf(data, params)
        likelihood: float = float(np.product(pdf_values))
        loglikelihood: float = float(np.sum(np.log(pdf_values)))
        num_data_points: int = len(data)
        num_params: int = len(params)
        aic: float = 2 * num_params - 2 * loglikelihood
        bic: float = -2 * loglikelihood + np.log(num_data_points) * num_params
        sse: float = float(np.sum((pdf_values - empirical_pdf_values) ** 2))

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
        }
        return self._FIT_TO(self, fit_info)

    def fit(self, empirical_fit: Callable, data: np.ndarray = None, params: tuple = None) -> Union[FittedContinuousUnivariate, FittedDiscreteUnivariate]:
        if data is not None:
            return self._fit_given_data(empirical_fit, data, params)
        elif params is not None:
            return self._fit_given_params(params)
        raise ValueError("data and/or params must be given in order to fit distribution.")

    @property
    def name(self) -> str:
        return self._name



def non_parametric_fit(data:np.ndarray):
    return ()


if __name__ == '__main__':
    import scipy.stats
    from sklarpy.univariate._fit import kde_fit
    normal = PreFitContinuousUnivariate('normal', scipy.stats.norm.pdf, scipy.stats.norm.cdf, scipy.stats.norm.ppf, scipy.stats.norm.support, scipy.stats.norm.fit, scipy.stats.norm.rvs)
    norm = normal.fit(params=(0, 1))
    gamma = PreFitContinuousUnivariate('gamma', scipy.stats.gamma.pdf, scipy.stats.gamma.cdf, scipy.stats.gamma.ppf, scipy.stats.gamma.support, scipy.stats.gamma.fit, scipy.stats.gamma.rvs)
    mydata = norm.rvs((1000,))
    fitgamma = gamma.fit(mydata)
    # norm.plot(show=False)
    # fitgamma.plot(show=False)

    poisson = PreFitDiscreteUnivariate('poisson', scipy.stats.poisson.pmf, scipy.stats.poisson.cdf, scipy.stats.poisson.ppf, scipy.stats.poisson.support, poisson_fit, scipy.stats.poisson.rvs)
    mydisdata = poisson.rvs((1000,), (6,))
    fitpoisson = poisson.fit(mydisdata)
    # fitpoisson.plot()
    print(fitgamma.gof())

    gaussian_kde = PreFitNumericalContinuousUnivariate('gaussian-kde', kde_fit)
    empirical = PreFitNumericalContinuousUnivariate('empirical', continuous_empirical_fit)
    discrete_empirical = PreFitNumericalDiscreteUnivariate('discrete-empirical', discrete_empirical_fit)